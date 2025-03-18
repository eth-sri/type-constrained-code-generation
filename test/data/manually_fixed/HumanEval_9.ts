function rollingMax(numbers: number[]): number[] {
    /**
     * From a given list of integers, generate a list of rolling maximum element found until given moment
     * in the sequence.
     * @example
     * rollingMax([1, 2, 3, 2, 3, 4, 2])
     * // returns [1, 2, 3, 3, 3, 4, 4]
     */

        // CHANGE: replace union number|null with number, introduce started variable
    let started = false;
    let runningMax: number = 0;
    let result: number[] = [];

    for (let n of numbers) {
        if (!started) {
            runningMax = n;
            started = true;
        } else {
            runningMax = Math.max(runningMax, n);
        }

        result.push(runningMax);
    }

    return result;
}
