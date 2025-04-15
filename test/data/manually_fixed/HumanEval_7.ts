function filterBySubstring(strings: string[], substring: string): string[] {
    /**
     * Filter an input list of strings only for ones that contain given substring
     *
     * Example usage:
     * filterBySubstring([], 'a'); // []
     * filterBySubstring(['abc', 'bacd', 'cde', 'array'], 'a'); // ['abc', 'bacd', 'array']
     */
    // CHANGE: add parens to filter
    return strings.filter((x) => x.includes(substring));
}
