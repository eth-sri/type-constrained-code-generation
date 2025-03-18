"""
Sample from the model, restricting to the acceptable vocabulary for the next token.

- EOS token is allowed iff parser is in accept state
- other tokens are allowed if they start with an admissable character

"""

from typing import Union, Literal

import fire
from frozendict import frozendict
from functools import partial
import stopit
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from termcolor import colored

from typesafe_llm.trie import Trie
from typesafe_llm.util import pflush, delete
from .parser.parser_ts import (
    custom_end_initial_state as custom_end_initial_state_ts,
    incremental_ts_parse,
)


def apply_mask(logits, mask):
    """
    Apply a mask to the logits (0/1)
    """
    if len(mask) < len(logits[0]):
        mask = torch.cat(
            [
                mask.to(logits.device),
                torch.zeros(len(logits[0]) - len(mask), dtype=torch.bool).to(
                    logits.device
                ),
            ]
        )
    elif len(mask) > len(logits[0]):
        mask = mask[: len(logits[0])]
    return logits.masked_fill(~mask.to(logits.device).unsqueeze(dim=0), -float("inf"))


def update_prefix_prompt(prompt, constrain_from, constrain_until):
    """
    Update the prefix and prompt based on the constraints
    i.e. determine whether to constrain, and where to constrain
    """
    if isinstance(constrain_from, str):
        assert constrain_from in prompt
        constrain = True
        index = prompt.index(constrain_from) + len(constrain_from)
        prefix = prompt[:index]
        result_program = prompt[index:]
        if isinstance(constrain_until, str) and constrain_until in result_program:
            constrain = False
            result_program = prefix + result_program
            prefix = ""
    else:
        constrain = False
        prefix = ""
        result_program = prompt

    return constrain, prefix, result_program


def true_next_token_mapping(token_id, tokenizer, prefix_ids):
    """
    The actual decoding result of a token id (as opposed to just checking the token dictionary)
    using the context and decoding mechanism.
    """
    input_ids = torch.cat([prefix_ids, token_id.unsqueeze(dim=0)], dim=1)
    new_out = tokenizer.decode(input_ids[0])
    prefix = tokenizer.decode(prefix_ids[0])
    diff = new_out[len(prefix) :]
    return diff


def batch_true_next_token_mapping(token_ids, tokenizer, prefix_ids):
    """
    The actual decoding result of a token id for the entire vocabulary,
    using the context and decoding mechanism.
    Leverages batching to speed up the process.
    """
    input_ids = torch.cat([prefix_ids.repeat(len(token_ids), 1), token_ids], dim=1)
    prefix = tokenizer.decode(prefix_ids[0])
    new_out = tokenizer.batch_decode(input_ids)
    diffs = []
    for i in range(len(token_ids)):
        diff = new_out[i][len(prefix) :]
        diffs.append(diff)
    return tuple(diffs)


def process_semicolon(prefix, prompt, states, initial_incremental_parse):
    """
    Automatically insert a semicolon if the LLM attempts to insert a newline 5 times
    """
    stripped = prompt.rstrip()
    suffix = prompt[len(stripped) :]
    if suffix.count("\n") >= 5:
        processed_prompt = stripped + ";"
        processed_states = initial_incremental_parse(processed_prompt)
        if len(processed_states) > 0:
            return True, prefix, processed_prompt, processed_states
        else:
            return False, prefix, prompt, states
    else:
        return False, prefix, prompt, states


def sample_constrained(
    device="mps",  # the device to load the model onto
    model_name="google/gemma-2-2b-it",
    prompt="""let a:num""",
    max_tokens=5,
    temperature=0.1,
    do_sample=False,
    constrain_from: Union[None, str] = None,
    constrain_until: str = None,
    trace=True,
    model=None,
    tokenizer=None,
    try_top_k=3,
    timeout=3000000,
    additional_identifiers=frozendict(),
    seed=None,
    reraise=False,
    semicolon=True,
    constrain_language: Literal["ts", "go"] = "ts",
    stop_on_end_constraint=True,
):
    """
    Sample from the model, restricting to the acceptable vocabulary for the next token.
    The restriction is implemented by checking the top-k tokens for validity and accepting the first valid token (in greedy mode) or sample k tokens until a valid token is found (in sampling mode).
    If no valid token is found, the entire vocabulary is checked for validity using the trie design by Poesia et al. (2021) and sampling from the masked logits.

    :param constrain_from: if str, constrain from that string, if None, do not constrain.
    :param constrain_until: if str, constrain until that string, after string model is unconstrained.
    :return: Resulting program, whether model finished with EOS, crashed exception, resamples
    """
    use_cache = not any(s in model_name for s in ("gemma-2-", "octocoder", "opt-"))
    if seed is not None:
        torch.manual_seed(seed)
    constrain, prefix, prompt = update_prefix_prompt(
        prompt, constrain_from, constrain_until
    )
    initial_until_count = prompt.count(constrain_until)
    # once we stop constraining, stop sampling if stop_on_end_constraint is True
    ever_constrained = constrain
    states = None
    _TRIE_CACHE = {}
    # list of (char index, k, p) where k is "how often did we resample to choose this token" and p is the predicted probability of the token
    # only for k>0
    resamples = []
    if constrain_language == "ts":
        incremental_parse = incremental_ts_parse
        initial_incremental_parse = partial(
            incremental_ts_parse,
            custom_end_initial_state_ts(constrain_until, additional_identifiers),
        )
    else:
        raise NotImplementedError(f"Unknown language {constrain_language}")
    with stopit.ThreadingTimeout(timeout):
        try:
            if not trace:

                def _pflush(*args, **kwargs):
                    pass
            else:
                _pflush = pflush
            _pflush("========== PROGRAM ==============\n")
            loading_str = "...loading model"
            if prefix:
                _pflush(prefix)
            if prompt:
                _pflush(prompt)
            _pflush(loading_str)

            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

            input_ids = (
                tokenizer([prefix + prompt], return_tensors="pt").to(device).input_ids
            )
            if use_cache:
                past = DynamicCache.from_legacy_cache(None)
            else:
                past = None

            delete(loading_str, _pflush)
            for _ in range(max_tokens):
                current_resample = None
                if states is None and constrain:
                    states = initial_incremental_parse(prompt)
                    if not states:
                        raise ValueError(
                            "initial prompt already rejected by sampler, invalid prompt"
                        )
                if constrain:
                    checking_str = f"({len(states)} states) ... "
                else:
                    checking_str = " ... "
                _pflush(checking_str)
                # compute model logits
                model_outputs = model(
                    input_ids, past_key_values=past, use_cache=use_cache
                )
                logits = model_outputs.logits[:, -1, :]
                orig_logits = logits.clone()
                if use_cache:
                    past = DynamicCache.from_legacy_cache(model_outputs.past_key_values)
                # else:
                #     past = model_outputs.past_key_values
                found = False
                best_choice = True
                if constrain:
                    eos_ok = any(s.accept for s in states)
                    if try_top_k > 0:
                        try_top_k = min(try_top_k, len(logits[-1]))
                        if not do_sample:
                            _, top_k_indices = torch.topk(logits, try_top_k)
                            probs = nn.functional.softmax(logits / temperature, dim=-1)
                        else:
                            probs = nn.functional.softmax(logits / temperature, dim=-1)
                            top_k_indices = torch.multinomial(
                                probs, num_samples=try_top_k
                            )
                        for i in range(try_top_k):
                            next_token_index = top_k_indices[:, i]
                            if i != 0:
                                best_choice = False
                                current_resample = (
                                    len(prefix) + len(prompt),
                                    i,
                                    probs[:, next_token_index.item()].item(),
                                )
                            if next_token_index.item() == tokenizer.eos_token_id:
                                if eos_ok:
                                    found = True
                                    break
                                continue
                            next_token_cand = true_next_token_mapping(
                                next_token_index,
                                tokenizer,
                                input_ids,
                            )
                            next_states = incremental_parse(
                                states,
                                next_token_cand,
                            )
                            if len(next_states) != 0:
                                found = True
                                states = next_states
                                break
                    if not found:
                        vocab_ids = sorted(tokenizer.get_vocab().values())
                        vocab = batch_true_next_token_mapping(
                            torch.tensor([[i] for i in vocab_ids], device=device),
                            tokenizer,
                            input_ids,
                        )
                        trie = _TRIE_CACHE.get(
                            vocab,
                        )
                        if trie is None:
                            # rebuild trie
                            trie = Trie.from_vocabulary(
                                vocab, enforce_token_maximality=False
                            )
                            _TRIE_CACHE[vocab] = trie
                        allowed_tokens = trie.antimonotonic_filter(
                            incremental_parse, states, _pflush=_pflush
                        )
                        mask = torch.zeros(len(vocab), dtype=torch.bool)
                        mask[[x[0] for x in allowed_tokens]] = True
                        mask[tokenizer.all_special_ids] = False
                        mask[tokenizer.eos_token_id] = eos_ok
                        logits = apply_mask(logits, mask)
                        if torch.all(logits == -float("inf")):
                            # no valid tokens
                            delete(checking_str, _pflush)
                            _pflush("==== INVALID PROBABILITIES =======\n")
                            return (
                                prefix + prompt + "INVALID PROBABILITIES",
                                True,
                                None,
                                resamples,
                            )
                if not found:
                    probs = nn.functional.softmax(logits / temperature, dim=-1)
                    if do_sample:
                        next_token_index = torch.multinomial(
                            probs, num_samples=1
                        ).squeeze(1)
                    else:
                        next_token_index = torch.argmax(logits, dim=-1)
                    if constrain:
                        current_resample = (
                            len(prefix) + len(prompt),
                            try_top_k + 1,
                            # compute the actual probability of the token
                            nn.functional.softmax(orig_logits / temperature, dim=-1)[
                                :, next_token_index.item()
                            ].item(),
                        )
                if current_resample is not None and constrain:
                    resamples.append(current_resample)

                next_token = true_next_token_mapping(
                    next_token_index, tokenizer, input_ids
                )
                if constrain and not found:
                    next_states = [
                        s
                        for i, (t, s) in enumerate(allowed_tokens)
                        if t == next_token_index
                    ][0]
                    states = next_states
                # note the model logits have larger dimension than available tokens but this should never be an issue
                # https://github.com/huggingface/transformers/issues/17431#issuecomment-1224231170
                if next_token_index.item() == tokenizer.eos_token_id:
                    # if the next token is eos, require that we accept
                    if not constrain or any(s.accept for s in next_states):
                        delete(checking_str, _pflush)
                        _pflush("==== MODEL EOS: TRUE =======\n")
                        return prefix + prompt, True, None, resamples
                delete(checking_str, _pflush)
                _pflush(next_token if best_choice else colored(next_token, "red"))
                prompt += next_token
                constrain, prefix, prompt = update_prefix_prompt(
                    prefix + prompt, constrain_from, constrain_until
                )
                ever_constrained = ever_constrained or constrain
                if ever_constrained and stop_on_end_constraint and not constrain:
                    _pflush("==== MODEL EOS: FALSE but CONSTRAIN ENDED =======\n")
                    return prefix + prompt, True, None, resamples
                # alternatively, end constraining after encountering \n```\n
                if (
                    constrain_from is None
                    and stop_on_end_constraint
                    and prompt.count(constrain_until) > initial_until_count
                ):
                    _pflush(
                        "==== MODEL EOS: FALSE but CONSTRAIN UNTIL ENCOUNTERED =======\n"
                    )
                    return prefix + prompt, True, None, resamples
                processed, processed_prefix, processed_prompt, processed_states = (
                    process_semicolon(prefix, prompt, states, initial_incremental_parse)
                )
                if semicolon and processed:
                    delete(prefix + prompt, _pflush)
                    prefix, prompt, states = (
                        processed_prefix,
                        processed_prompt,
                        processed_states,
                    )
                    _pflush(prefix + prompt)
                    input_ids = (
                        tokenizer([prefix + prompt], return_tensors="pt")
                        .to(device)
                        .input_ids
                    )
                    if use_cache:
                        past = DynamicCache.from_legacy_cache(None)
                    else:
                        past = None
                else:
                    if not use_cache:
                        input_ids = torch.cat(
                            [input_ids, next_token_index.unsqueeze(dim=0)], dim=1
                        )
                    else:
                        input_ids = next_token_index.unsqueeze(dim=0)
            _pflush("==== MODEL EOS: FALSE =======\n")
            return prefix + prompt, False, None, resamples
        except Exception as e:
            if reraise:
                raise e
            return prefix + prompt, False, e, resamples
    # only way to get here is through timeout
    return prefix + prompt, False, TimeoutError("Sampling timed out"), resamples


def sample_n_constrained(
    n: int,
    device="cuda",  # the device to load the model onto
    model_name="facebook/opt-350m",
    prompt="""let a:num""",
    max_tokens=5,
    temperature=0.1,
    do_sample=False,
    constrain_from: Union[None, str] = None,
    constrain_until: str = None,
    trace=True,
    model=None,
    tokenizer=None,
    try_top_k=3,
    timeout=300,
    additional_identifiers=frozendict(),
    initial_seed=None,
):
    """
    Same as sample_constrained but samples n times
    """
    res = []
    for i in range(n):
        seed = initial_seed + i if initial_seed is not None else None
        result = sample_constrained(
            device=device,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            constrain_from=constrain_from,
            constrain_until=constrain_until,
            trace=trace,
            model=model,
            tokenizer=tokenizer,
            try_top_k=try_top_k,
            timeout=timeout,
            additional_identifiers=additional_identifiers,
            seed=seed,
        )
        res.append(result)
    return res


if __name__ == "__main__":
    fire.Fire(sample_constrained)
