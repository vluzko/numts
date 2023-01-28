import {tensor} from '../tensor';
import {_div} from './binary_ops';

/**
 * Return the index of the minimum value.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to take the min over.
 * @return {tensor | number}
 */
export function _argmin(a: tensor, axis?: number): tensor | number {
    const min_index = (...args) => {
        let min = args[0];
        let index = 0;
        for (let i = 1; i < args.length; i++) {
            if (args[i] < min) {
                min = args[i];
                index = i;
            }
        }
        return index;
    };
    return a.apply_to_axis(e => min_index(...e), axis);
}

/**
 * Return the index of the maximum value.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to take the max over.
 * @return {tensor | number}
 */
export function _argmax(a: tensor, axis?: number): tensor | number {
    const max_index = (...args) => {
        let max = args[0];
        let index = 0;
        for (let i = 1; i < args.length; i++) {
            if (args[i] > max) {
                max = args[i];
                index = i;
            }
        }
        return index;
    };
    return a.apply_to_axis(e => max_index(...e), axis);
}

/**
 * Return true if all elements are true.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to take the and over.
 * @return {tensor | number}
 */
export function _all(a: tensor, axis?: number): tensor | number {
    const f = data => {
        for (let value of data) {
            if (!value) {
                return false;
            }
        }
        return true;
    }
    return a.apply_to_axis(f, axis);
}

/**
 * Return true if any element is true.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to take the or over.
 * @return {tensor | number}
 */
export function _any(a: tensor, axis?: number): tensor | number {
    const f = data => {
        for (let value of data) {
            if (value) {
                return true;
            }
        }
        return false;
    }
    return a.apply_to_axis(f, axis);
}

/**
 * Returns the maximum element of the array.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to take the max over.
 * @return {tensor | number}
 */
export function _max(a: tensor, axis?: number): tensor | number {
    return a.apply_to_axis(e => Math.max(...e), axis);
}

/**
 * Returns the minimum element of the array along the specified axis.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to take the min over.
 * @return {tensor | number}
 */
export function _min(a: tensor, axis?: number): tensor | number {
    return a.apply_to_axis(e => Math.min(...e), axis);
}

/**
 * Sum the entries of the array along the specified axis.
 * @param {tensor} a - The tensor.
 * @param {number} axis - The axis to sum over.
 * @return {tensor | number}
 */
export function _sum(a: tensor, axis?: number): tensor | number {
    return a.reduce((a, e) => a + e, 0, axis);
}
