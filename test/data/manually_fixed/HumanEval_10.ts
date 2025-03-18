function isPalindrome(str: string): boolean {
    /** Test if given string is a palindrome */
    return str === str.split('').reverse().join('');
}

function makePalindrome(str: string): string {
    /** Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    */

    if (!str) {
        return '';
    }

    let beginningOfSuffix = 0;

    while (!isPalindrome(str.substring(beginningOfSuffix))) {
        beginningOfSuffix += 1;
    }

    // CHANGE: remove console.log
    return str + str.substring(0, beginningOfSuffix).split('').reverse().join('');
}