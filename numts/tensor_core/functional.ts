import {tensor, TypedArray} from '../tensor';
import * as constructors from './constructors';
import {indexing} from '../indexing';


/**
 * Map the array.
 * @param f
 * @param {number} axis
 * @return {tensor}
 */
export function map(f, axis?: number): tensor {
    const new_data = this.data.map(f);
    return constructors.array(new_data, this.shape, { disable_checks: true, dtype: this.dtype })
}


/**
 * Accumulating map over the entire array or along a particular axis.
 * If no axis is provided a flat array is returned.
 * Otherwise the shape of the result is the same as the shape of the original array.
 * @param f - Function to use.
 * @param {number} axis - Axis to map over.
 * @param {number} start  - Initial value.
 * @param {string} dtype  - Dtype of the result array.
 * @return {tensor | number}
 */
export function accum_map(f, axis?: number, start?: number, dtype?: string): tensor | number {
    dtype = dtype === undefined ? this.dtype : dtype;
    let new_array;
    if (axis === undefined) {
        // TODO: Views: Use size of view.

        new_array = constructors.zeros(this.length, dtype);
        let first_value;

        if (start !== undefined) {
            new_array.data[0] = start;
        }

        let previous_index = 0;
        let index_in_new = 0;
        for (let index of this._iorder_data_iterator()) {
            new_array.data[index_in_new] = f(new_array.data[previous_index], this.data[index]);
            previous_index = index_in_new;
            index_in_new += 1;
        }

    } else {
        const [lower, upper, steps] = this._slice_for_axis(axis);
        new_array = constructors.zeros(this.shape, dtype);
        const step_along_axis = this.stride[axis];

        for (let index of this._iorder_data_iterator(lower, upper, steps)) {
            let first_value;

            if (start !== undefined) {
                first_value = f(start, this.data[index]);
            } else {
                first_value = this.data[index];
            }

            new_array.data[index] = first_value;
            let previous_index = index;
            for (let i = 1; i < this.shape[axis]; i++) {
                const new_index = index + i * step_along_axis;
                new_array.data[new_index] = f(new_array.data[previous_index], this.data[new_index]);
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
export function apply_to_axis(f: (a: TypedArray | number[]) => any, axis?: number, dtype?: string): tensor | number {
    dtype = dtype === undefined ? this.dtype : dtype;
    if (axis === undefined) {
        return f(this.data);
    } else {
        const new_shape = indexing.new_shape_from_axis(this.shape, axis);
        let new_array = constructors.zeros(new_shape, dtype);
        const step_along_axis = this.stride[axis];
        for (let [old_index, new_index] of this.map_old_indices_to_new(axis)) {
            let axis_values = [];
            for (let i = 0; i < this.shape[axis]; i++) {
                axis_values.push(this.data[old_index + i * step_along_axis]);
            }

            new_array.data[new_index] = f(axis_values);
        }

        return new_array;
    }
}

/**
 * Reduce the array over the specified axes with the specified function.
 * @param {(number, number, number?, array?) => number} f
 * @param {number} axis
 * @param {string} dtype
 */
export function reduce(f: (accum: number, e: number, i?: number, array?) => number, initial?: number, axis?: number, dtype?: string): number | tensor {
    dtype = dtype === undefined ? this.dtype : dtype;
    if (axis === undefined) {
        const iter = this._iorder_value_iterator()[Symbol.iterator]();
        // Deal with initial value
        let { done, value } = iter.next();
        // If it's an empty array, return.
        let accum;
        if (done) {
            return this;
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
        const new_shape = indexing.new_shape_from_axis(this.shape, axis);
        let new_array = constructors.zeros(new_shape, dtype);
        const step_along_axis = this.stride[axis];
        for (let [old_index, new_index] of this.map_old_indices_to_new(axis)) {
            let accum = initial === undefined ? this.data[old_index] : f(initial, this.data[old_index]);
            for (let i = 1; i < this.shape[axis]; i++) {
                accum = f(accum, this.data[old_index + i * step_along_axis]);
            }

            new_array.data[new_index] = accum;
        }
        return new_array;
    }
}