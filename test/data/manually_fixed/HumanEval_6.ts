function parseNestedParens(parenString: string): number[] {
    /**
     * Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
     * For each of the group, output the deepest level of nesting of parentheses.
     * E.g. (()()) has maximum two levels of nesting while ((())) has three.
     *
     * >>> parseNestedParens('(()()) ((())) () ((())()())')
     * [2, 3, 1, 3]
     */

    function parseParenGroup(s: string): number {
        let depth = 0;
        let maxDepth = 0;
        for (const c of s) {
            if (c === '(') {
                depth += 1;
                maxDepth = Math.max(depth, maxDepth);
            } else {
                depth -= 1;
            }
        }
        return maxDepth;
    }
    // CHANGE: rewrite map expression
    let result: number[] = [];
    for (const group of parenString.split(' ', 100000)) {
        if (group) {
            result.push(parseParenGroup(group));
        }
    }
    return result;
}
