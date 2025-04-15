function separateParenGroups(parenString: string): string[] {
    /**
     * Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
     * separate those group into separate strings and return the list of those.
     * Separate groups are balanced (each open brace is properly closed) and not nested within each other
     * Ignore any spaces in the input string.
     *
     * Example:
     * console.log(separateParenGroups('( ) (( )) (( )( ))'))
     * // Output: ['()', '(())', '(()())']
     */

    const result: string[] = [];
    let currentString: string[] = [];
    let currentDepth = 0;

    for (const c of parenString) {
        if (c === '(') {
            currentDepth += 1;
            currentString.push(c);
        } else {
            // CHANGE: fix nesting
            if (c === ')') {
                currentDepth -= 1;
                currentString.push(c);

                if (currentDepth === 0) {
                    result.push(currentString.join(''));
                    currentString = [];
                }
            }
        }
    }

    return result;
}