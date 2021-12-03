import {tensor, TypedArray} from '../tensor';
import * as constructors from './constructors';
import {indexing} from './indexing';

/**
 * Accumulating map over the entire array or along a particular axis.
 * If no axis is provided a flat array is returned.
 * Otherwise the shape of the result is the same as the shape of the original array.
 * @param f - function to use.
 * @param {number} axis - Axis to map over.
 * @param {number} start  - Initial value.
 * @param {string} dtype  - Dtype of the result array.
 * @return {tensor | number}
 */
export function _accum_map(a: tensor, f, axis?: number, start?: number, dtype?: string): tensor | number {
    dtype = dtype === undefined ? a.dtype : dtype;
    let new_array;
    if (axis === undefined) {
        // TODO: Views: Use size of view.

        new_array = constructors.zeros(a.length, dtype);

        if (start !== undefined) {
            new_array.data[0] = start;
        }

        let previous_index = 0;
        let index_in_new = 0;
        for (let index of a._iorder_data_iterator()) {
            new_array.data[index_in_new] = f(new_array.data[previous_index], a.data[index]);
            previous_index = index_in_new;
            index_in_new += 1;
        }
    } else {
        const [lower, upper, steps] = a._slice_for_axis(axis);
        new_array = constructors.zeros(a.shape, dtype);
        const step_along_axis = a.stride[axis];

        for (let index of a._iorder_data_iterator(lower, upper, steps)) {
            let first_value;

            if (start !== undefined) {
                first_value = f(start, a.data[index]);
            } else {
                first_value = a.data[index];
            }

            new_array.data[index] = first_value;
            let previous_index = index;
            for (let i = 1; i < a.shape[axis]; i++) {
                const new_index = index + i * step_along_axis;
                new_array.data[new_index] = f(new_array.data[previous_index], a.data[new_index]);
                previous_index = new_index;
            }
        }
    }
    return new_array;
}

/**
 * Apply the given function along the given axis.
 * @param {(a: (TypedArray | number[])) => any} f
 * @param {number} axis
 * @param {string} dtype
 * @return {tensor | number}
 */
export function _apply_to_axis(a: tensor, f: (a: TypedArray | number[]) => any, axis?: number, dtype?: string): tensor | number {
    dtype = dtype === undefined ? a.dtype : dtype;
    if (axis === undefined) {
        return f(a.data);
    } else {
        const new_shape = indexing.new_shape_from_axis(a.shape, axis);
        let new_array = constructors.zeros(new_shape, dtype);
        const step_along_axis = a.stride[axis];
        for (let [old_index, new_index] of a.map_old_indices_to_new(axis)) {
            let axis_values = [];
            for (let i = 0; i < a.shape[axis]; i++) {
                axis_values.push(a.data[old_index + i * step_along_axis]);
            }

            new_array.data[new_index] = f(axis_values);
        }

        return new_array;
    }
}

/**
 * Map the array.
 * @param f
 * @return {tensor}
 */
export function _map(a: tensor, f): tensor {
    const new_data = a.data.map(f);
    return constructors.array(new_data, a.shape, { disable_checks: true, dtype: a.dtype })
}

/**
 * Reduce the array over the specified axes with the specified function.
 * @param {(number, number, number?, array?) => number} f
 * @param {number} axis
 * @param {string} dtype
 */
export function _reduce(a: tensor, f: (accum: number, e: number, i?: number, array?) => number, initial?: number, axis?: number, dtype?: string): number | tensor {
    dtype = dtype === undefined ? a.dtype : dtype;
    if (axis === undefined) {
        const iter = a._iorder_value_iterator()[Symbol.iterator]();
        // Deal with initial value
        let { done, value } = iter.next();
        // If it's an empty array, return.
        let accum;
        if (done) {
            return a;
        } else {
            accum = initial === undefined ? value : f(initial, value);
            while (true) {
                let { done, value } = iter.next();
                if (done) {
                    break;
                } else {
                    accum = f(accum, value);
                }
            }
            return accum;
        }
    } else {
        const new_shape = indexing.new_shape_from_axis(a.shape, axis);
        let new_array = constructors.zeros(new_shape, dtype);
        const step_along_axis = a.stride[axis];
        for (let [old_index, new_index] of a.map_old_indices_to_new(axis)) {
            let accum = initial === undefined ? a.data[old_index] : f(initial, a.data[old_index]);
            for (let i = 1; i < a.shape[axis]; i++) {
                accum = f(accum, a.data[old_index + i * step_along_axis]);
            }

            new_array.data[new_index] = accum;
        }
        return new_array;
    }
}