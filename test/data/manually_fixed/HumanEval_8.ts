// CHANGE: replace tuple type with list
function sumProduct(numbers: number[]): number[] {
    /**
     * For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
     * Empty sum should be equal to 0 and empty product should be equal to 1.
     * sumProduct([]) => [0, 1]
     * sumProduct([1, 2, 3, 4]) => [10, 24]
     */

    let sumValue = 0;
    let prodValue = 1;

    for (const n of numbers) {
        sumValue += n;
        prodValue *= n;
    }
    return [sumValue, prodValue];
}
