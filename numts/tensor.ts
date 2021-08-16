import { utils } from './utils';
import { indexing } from './indexing';
import * as arithmetic from './tensor_core/arithmetic';
import new_shape_from_axis = indexing.new_shape_from_axis;

type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;
type Numeric = TypedArray | number[];
export type Broadcastable = number | TypedArray | tensor | number[];
type Shape = number[] | Uint32Array;

interface NumericalArray {
    byteLength;
    map;
    slice;
    reduce: (cb: (previousValue: number, currentValue: number, currentIndex: number, array: NumericalArray) => number) => number;

    new(number): NumericalArray;
}

interface ArrayOptions {
    dtype?: string
    disable_checks?: boolean
}

namespace errors {
    export class MismatchedSizes extends Error {
        constructor() {
            super('Array sizes do not match.')
        }
    }
    export class MismatchedShapes extends Error {
        constructor() {
            super('Array shapes do not match.')
        }
    }
    export class BadData extends Error {
        constructor() {
            super('Bad data.');
        }
    }
    export class DataNotArrayError extends Error { }
    export class DataNullOrNotNumeric extends Error { }
    export class BadShape extends Error { }
    export class MismatchedShapeSize extends Error { }
    export class WrongIterableSize extends Error { }
    export class NestedArrayHasInconsistentDimensions extends Error { }
}

export class tensor {

    public data;
    readonly offset: Uint32Array;
    readonly stride: Uint32Array;
    readonly dstride: Uint32Array;
    public initial_offset: number;
    public shape: Uint32Array;
    public length: number;
    public dtype: string;
    public is_view: boolean;

    /**
     *
     * @param data
     * @param {Uint32Array} shape     - The shape of the array.
     * @param {Uint32Array} offset    - The offset of the array from the start of the underlying data.
     * @param {Uint32Array} stride    - The stride of the array.
     * @param {Uint32Array} dstride   - The stride of the underlying data.
     * @param {number} size           - The number of elements in the array.
     * @param {string} dtype          -
     * @param {boolean} is_view       -
     * @param {number} initial_offset -
     * @constructor
     */
    private constructor(data,
        shape: Uint32Array,
        offset: Uint32Array,
        stride: Uint32Array,
        dstride: Uint32Array,
        size: number,
        dtype?: string,
        is_view?: boolean,
        initial_offset?: number) {
        this.shape = shape;
        this.offset = offset;
        this.stride = stride;
        this.length = size;
        this.dstride = dstride;
        if (dtype !== undefined) {
            const array_type = utils.dtype_map(dtype);
            if (!(data instanceof array_type)) {
                this.data = new array_type(data);
            } else {
                this.data = data;
            }
            this.dtype = dtype;
        } else {
            this.data = data;
            this.dtype = 'float64';
        }
        this.initial_offset = initial_offset === undefined ? 0 : initial_offset;
        this.is_view = is_view === undefined ? false : is_view;
    }
    argmax() { }

    argmin() { }

    argpartition() { }

    argsort() { }

    /**
     * Return a copy of the tensor cast to the specified type.
     */
    as_type(dtype: string): tensor {
        const array_type = utils.dtype_map(dtype);
        const new_data = new array_type(this.data.slice(0));
        return new tensor(new_data, this.shape.slice(0), this.offset.slice(0), this.stride.slice(0), this.dstride.slice(0), this.length, dtype, this.is_view, this.initial_offset);
    }

    /**
     * Clip all values in the array to be in the specified range.
     * @param lower - The lower bound of the range.
     * @param upper - The upper bound of the range.
     */
    clip(lower: number, upper: number): tensor {
        return this.map(e => {
            if (e < lower) {
                return lower;
            } else if (e > upper) {
                return upper;
            } else {
                return e;
            }
        });
    }

    /**
     * The cumulative product along the given axis.
     * @param {number} axis
     * @param {string} dtype
     * @return {number | tensor}
     */
    cumprod(axis?: number, dtype?: string): number | tensor {
        return this.accum_map((acc, b) => acc * b, axis, 1, dtype);
    }

    /**
     * The cumulative sum of the array along the given axis.
     * @param {number} axis
     * @param {string} dtype
     */
    cumsum(axis?: number, dtype?: string): number | tensor {
        return this.accum_map((acc, b) => acc + b, axis, undefined, dtype);
    }

    diagonal() { }

    dot(b: tensor) {
        return tensor.dot(this, b);
    }

    /**
     * Fill the array with the given value, in-place.
     * @param {number} value  - The value to fill the array with
     * @return {tensor}     - The filled array.
     */
    fill(value: number) {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = value;
        }
        return this;
    }

    //#region METHOD CONSTRUCTORS

    /**
     * Create a copy of this with a different shape.
     * @param {Uint32Array} new_shape - The shape to make the new array.
     * @return {tensor}             - The reshaped array.
     */
    reshape(...new_shape: Array<Uint32Array | number[] | number>): tensor {
        let shape: Uint32Array | number[];
        if (utils.is_numeric_array(new_shape[0])) {
            // @ts-ignore
            shape = new_shape[0];
        } else {
            // @ts-ignore
            shape = new_shape;
        }

        if (Array.isArray(shape)) {
            shape = new Uint32Array(shape);
        }

        const new_size = indexing.compute_size(shape);
        const size = indexing.compute_size(this.shape);
        if (size !== new_size) {
            throw new errors.BadShape(`Array cannot be reshaped because sizes do not match. Size of underlying array: ${size}. Size of reshaped array: ${shape}`);
        }
        let value_iter = this._iorder_value_iterator();
        return tensor.from_iterable(value_iter, shape, this.dtype);
    }

    /**
     * Flatten an array. Elements will be in iteration order.
     * @returns - The flattened array
     */
    flatten(): tensor {
        const shape = new Uint32Array([this.length]);
        return tensor.from_iterable(this._iorder_value_iterator(), shape, this.dtype);
    }

    /**
     * Returns the negation of this array.
     */
    neg(): tensor {
        const new_data = this.data.map(x => -x);
        return tensor.array(new_data, this.shape, { disable_checks: true, dtype: this.dtype });
    }

    /**
     * Return the transpose of this array.
     */
    transpose(): tensor {
        const new_shape = this.shape.slice(0).reverse();
        let new_array = tensor.zeros(new_shape, this.dtype);
        for (let index of this._iorder_index_iterator()) {
            const value = this.g(...index);
            const new_index = index.reverse();
            new_array.s(value, ...new_index);
        }
        return new_array;
    }

    /**
     * Extract the upper triangle of this tensor.
     */
    triu(): tensor {
        const iter = utils.imap(this._iorder_index_iterator(), i => {

            if (i[i.length - 2] <= i[i.length - 1]) {
                return this.g(...i);
            } else {
                return 0;
            }
        });
        return tensor.from_iterable(iter, this.shape, this.dtype);
    }

    /**
     * Extract the lower triangle of this tensor.
     */
    tril(): tensor {
        const iter = utils.imap(this._iorder_index_iterator(), i => {

            if (i[i.length - 2] >= i[i.length - 1]) {
                return this.g(...i);
            } else {
                return 0;
            }
        });
        return tensor.from_iterable(iter, this.shape, this.dtype);
    }

    //#endregion METHOD CONSTRUCTORS

    //#region AGGREGATION

    /**
     * Return true if all elements are true.
     */
    all(axis?: number): number | tensor {
        const f = data => {
            for (let value of data) {
                if (!value) {
                    return false;
                }
            }
            return true;
        }
        return this.apply_to_axis(f, axis);
    }

    /**
     * Return true if any element is true.
     */
    any(axis?: number): number | tensor {
        const f = data => {
            for (let value of data) {
                if (value) {
                    return true;
                }
            }
            return false;
        }
        return this.apply_to_axis(f, axis);
    }

    /**
     * Returns the maximum element of the array.
     * @param {number} axis
     * @return {number}
     */
    max(axis?: number): tensor | number {
        return this.apply_to_axis(e => Math.max(...e), axis);
    }

    /**
     * Returns the minimum element of the array along the specified axis.
     * @param {number} axis
     * @return {number}
     */
    min(axis?: number): tensor | number {
        return this.apply_to_axis(e => Math.min(...e), axis);
    }

    /**
     * Calculate the mean of the array.
     * @param {number} axis
     */
    mean(axis?: number): tensor | number {
        if (axis === undefined) {
            return <number>this.sum() / this.length;
        } else {
            return arithmetic._div(this.sum(axis), this.shape[axis]);
        }
    }

    /**
     * Return the standard deviation along the specified axis.
     * @param {number} axis
     * @return {number}
     */
    stdev(axis?: number): tensor | number {
        const mean = this.mean(axis);
        const squared_values = this.power(2);
        const mean_of_squares = squared_values.mean(axis);
        const squared_mean = arithmetic._power(mean, 2);
        const difference = arithmetic._sub(mean_of_squares, squared_mean);
        const result = arithmetic._power(difference, 0.5);
        if (axis === undefined) {
            return result.data[0];
        } else {
            return result;
        }
    }

    /**
     * Return the variance along the specified axis.
     * @param {number} axis
     * @return {tensor | number}
     */
    variance(axis?: number): tensor | number {
        const std = this.stdev(axis);
        const result = arithmetic._power(std, 0.5);
        if (axis === undefined) {
            return result.data[0];
        } else {
            return result;
        }
    }

    /**
     * Sum the entries of the array along the specified axis.
     * @param {number} axis
     * @return {number}
     */
    sum(axis?: number): number | tensor {
        return this.reduce((a, e) => a + e, 0, axis);
    }

    // #endregion AGGREGATION

    //#region FUNCTIONAL

    /**
     * Map the array.
     * @param f
     * @param {number} axis
     * @return {tensor}
     */
    map(f, axis?: number): tensor {
        const new_data = this.data.map(f);
        return tensor.array(new_data, this.shape, { disable_checks: true, dtype: this.dtype })
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
    accum_map(f, axis?: number, start?: number, dtype?: string): tensor | number {
        dtype = dtype === undefined ? this.dtype : dtype;
        let new_array;
        if (axis === undefined) {
            // TODO: Views: Use size of view.

            new_array = tensor.zeros(this.length, dtype);
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
            new_array = tensor.zeros(this.shape, dtype);
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
    apply_to_axis(f: (a: TypedArray | number[]) => any, axis?: number, dtype?: string): tensor | number {
        dtype = dtype === undefined ? this.dtype : dtype;
        if (axis === undefined) {
            return f(this.data);
        } else {
            const new_shape = indexing.new_shape_from_axis(this.shape, axis);
            let new_array = tensor.zeros(new_shape, dtype);
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
    reduce(f: (accum: number, e: number, i?: number, array?) => number, initial?: number, axis?: number, dtype?: string): number | tensor {
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
            let new_array = tensor.zeros(new_shape, dtype);
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

    //#endregion FUNCTIONAL

    /**
     * Returns the indices of the nonzero elements of the array.
     */
    nonzero(): Uint32Array[] {
        let indices = [];
        const steps = utils.fixed_ones(this.shape.length);
        for (let index of indexing.iorder_index_iterator(this.offset, this.shape, steps)) {
            const real_value = this._compute_real_index(index);
            if (this.data[real_value] !== 0) {
                indices.push(index)
            }
        }
        return indices
    }

    partition() { }

    /**
     * Compute an element-wise power.
     * @param {number} exp
     */
    power(exp: number) {
        return this.map(e => Math.pow(e, exp));
    }

    prod() { }

    round() { }

    sort() { }

    squeeze() { }

    trace() { }

    /**
     * Drop any dimensions that equal 1.
     * @return {tensor}
     */
    drop_unit_dimensions(): tensor {
        // Drop extra dimensions.
        let flattened_shape = [];
        let flattened_stride = [];
        let flattened_offset = [];
        this.shape.forEach((e, i) => {
            if (e > 1) {
                flattened_shape.push(e);
                flattened_stride.push(this.stride[i]);
                flattened_offset.push(this.offset[i]);
            }
        });
        const size = indexing.compute_size(flattened_shape);
        const new_shape = new Uint32Array(flattened_shape);
        const new_offset = new Uint32Array(flattened_offset);
        const new_stride = new Uint32Array(flattened_stride);

        const view = new tensor(this.data, new_shape, new_offset, new_stride, new_stride, size, this.dtype, true);

        return view;
    }

    /**
     * Return a slice of an array. Copies the underlying data.
     * @param indices
     */
    c_slice(...indices) {

    }

    /**
     * Return a slice of an array. Does not copy the underlying data. Does not drop dimensions.
     * @param indices - The indices to slice on. Can be either a single array / TypedArray, or a spread of integers.
     *                  
     * 
     * @example
     *    let a = numts.arange(24).reshape(2, 3, 4).slice(0); // a is the [0, :, :] slice.
     * @example
     *    let b = numts.arange(24).reshape(2, 3, 4).slice([2, 3]); // b is the [2:3, :, :] slice.
     * @example
     *    let b = numts.arange(24).reshape(2, 3, 4).slice(2, 3); // b is the [2, 3, :] slice.
     *   
     */
    slice(...indices: Array<number | number[]>): tensor {

        // Handle empty inputs.
        // @ts-ignore
        if (indices.length === 1 && !utils.is_numeric(indices[0]) && indices[0].length === 0) {
            return this;
        }
        const positive_indices = indexing.convert_negative_indices(indices, this.shape);
        let start = new Uint32Array(this.shape.length);
        let end = this.shape.slice();
        let steps = new Uint32Array(this.shape.length);
        let dims_to_drop = new Set();

        steps.fill(1);
        let initial_offset = this.initial_offset;
        let i = 0;
        for (let index of positive_indices) {
            if (index === null) {

            } else if (utils.is_numeric(index)) {
                start[i] = index;
                end[i] = index + 1;
                dims_to_drop.add(i);
                // initial_offset += index * this.dstride[i];
            } else if (index.length === 2) {
                start[i] = index[0];
                end[i] = index[1];
            } else if (index.length === 3) {
                start[i] = index[0];
                end[i] = index[1];
                steps[i] = index[2];
            } else {
                throw new Error(`Arguments to slice were wrong: ${positive_indices}. Broke on ${index}.`);
            }
            i += 1;
        }

        const new_shape = indexing.new_shape_from_slice(start, end, steps);
        const size = indexing.compute_size(new_shape);

        const offset = start.map((e, j) => e + this.offset[j]);
        const stride = steps.map((e, j) => e * this.stride[j]);
        initial_offset += start.reduce((acc, e, j) => acc + e * this.stride[j], 0);

        const filt = (e, j) => !dims_to_drop.has(j);

        const new_stride = stride.filter(filt);
        const new_dstride = this.dstride.filter(filt);
        const view = new tensor(this.data,
            new_shape.filter(filt),
            offset.filter(filt),
            new_stride, new_dstride, size, this.dtype, true, initial_offset);
        return view;
    }

    /**
     * Get the value at the given index.
     * @param indices
     * @return {number}
     */
    g(...indices): number {
        if (indices.length !== this.shape.length) {
            throw new Error(`Need more dimensions.`)
        }
        const positive_indices = indexing.convert_negative_indices(indices, this.shape);
        const real_index = this._compute_real_index(positive_indices);
        return this.data[real_index];
    }

    /**
     * Set an element of the array.
     * @param values
     * @param indices
     */
    s(values: Broadcastable, ...indices) {
        // Set a single element of the array.
        if (indexing.checks_indices_are_single_index(...indices) && indices.length === this.shape.length) {
            if (!utils.is_numeric(values)) {
                throw new Error(`To set a single element of the array, the values must be a scalar. Got ${values}.`);
            }
            const positive_indices = indexing.convert_negative_indices(indices, this.shape);
            const real_index = this._compute_real_index(positive_indices);
            this.data[real_index] = values;
            return;
        }

        const view = this.slice(...indices);

        let b_array = tensor._upcast_to_tensor(values);

        // Check that shapes are compatible.
        const difference = view.shape.length - b_array.shape.length;
        if (difference < 0) {
            throw new Error(`Bad dimensions for broadcasting. a: ${view.shape}, b: ${b_array.shape}`);
        }

        for (let i = 0; i < b_array.shape.length; i++) {
            if (b_array.shape[i] !== view.shape[i + difference] && b_array.shape[i] !== 1) {
                throw new Error(`Bad dimensions for broadcasting. a: ${view.shape}, b: ${b_array.shape}`);
            }
        }
        const iterator = utils.zip_longest(view._iorder_data_iterator(), b_array._iorder_data_iterator());

        for (let [a_index, b_index] of iterator) {

            view.data[a_index] = b_array.data[b_index];
        }
    }

    //#region Indexing

        /**
         * Computes the index of a value in the underlying data array based on a passed index.
         * @param indices
         * @return {number} - The index
         * @private
         */
        _compute_real_index(indices): number {
            return indexing.index_in_data(indices, this.stride, this.initial_offset);
        }

        /**
         * Compute lower, upper, and steps for a slice of an array along `axis`.
         * @param {number} axis
         * @return {[Uint32Array, Uint32Array, Uint32Array]}  - [lower, upper, steps]
         * @private
         */
        private _slice_for_axis(axis: number): [Uint32Array, Uint32Array, Uint32Array] {
            const lower = new Uint32Array(this.shape.length);
            let upper = this.shape.slice(0);
            const steps = utils.fixed_ones(this.shape.length);
            upper[axis] = 1;
            return [lower, upper, steps];
        }

        /**
         * Return an iterator over real indices of the old array and real indices of the new array.
         * @param {number} axis
         * @return {Iterable<number[]>}
         * @private
         */
        private map_old_indices_to_new(axis: number): Iterable<number[]> {
            const new_shape = indexing.new_shape_from_axis(this.shape, axis);
            let new_array = tensor.zeros(new_shape, this.dtype);

            let [lower, upper, steps] = this._slice_for_axis(axis);

            let old_iter = this._iorder_data_iterator(lower, upper, steps)[Symbol.iterator]();
            let new_iter = new_array._iorder_data_iterator()[Symbol.iterator]();
            return utils.zip_iterable(old_iter, new_iter);
        }

        /**
         * Create an iterator over the data indices of the elements of the tensor, in index order.
         * Just a convenience wrapper around `indexing.iorder_data_iterator`.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         * @return {Iterable<number>}
         */
        _iorder_data_iterator(lower_or_upper?: Uint32Array, upper_bounds?: Uint32Array, steps?: Uint32Array): Iterable<number> {
            const bounds = this._calculate_slice_bounds(lower_or_upper, upper_bounds, steps);
            return indexing.iorder_data_iterator(bounds[0], bounds[1], bounds[2], this.stride, this.initial_offset);
        }

        /**
         * Create an iterator over the indices of the elements of the tensor, in index order.
         * Just a convenience wrapper around `indexing.iorder_index_iterator`.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         * @return {Iterable<number>}
         */
        _iorder_index_iterator(lower_or_upper?: Uint32Array, upper_bounds?: Uint32Array, steps?: Uint32Array): Iterable<Uint32Array> {
            const bounds = this._calculate_slice_bounds(lower_or_upper, upper_bounds, steps);
            return indexing.iorder_index_iterator(...bounds);
        }

        /**
         * Create an iterator over the values of the array, in index order.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         * @private
         */
        _iorder_value_iterator(lower_or_upper?: Uint32Array, upper_bounds?: Uint32Array, steps?: Uint32Array): Iterable<number> {

            const index_iterator = this._iorder_data_iterator(lower_or_upper, upper_bounds, steps);
            const self = this;
            const iter = {
                [Symbol.iterator]: function* () {
                    for (let index of index_iterator) {
                        yield self.data[index];
                    }
                }
            }

            return iter;
        }

        /**
         * Create an iterator over the data indices of the elements of the tensor, in data order.
         * Just a convenience wrapper around `indexing.dorder_data_iterator`.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         * @return {Iterable<number>}
         */
        _dorder_data_iterator(lower_or_upper?: Uint32Array, upper_bounds?: Uint32Array, steps?: Uint32Array): Iterable<number> {
            const bounds = this._calculate_slice_bounds(lower_or_upper, upper_bounds, steps);
            return indexing.dorder_data_iterator(bounds[0], bounds[1], bounds[2], this.stride, this.initial_offset);
        }

        /**
         * Create an iterator over the indices of the elements of the tensor, in data order.
         * Just a convenience wrapper around `indexing.dorder_index_iterator`.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         * @return {Iterable<number>}
         */
        _dorder_index_iterator(lower_or_upper?: Uint32Array, upper_bounds?: Uint32Array, steps?: Uint32Array): Iterable<Uint32Array> {
            const bounds = this._calculate_slice_bounds(lower_or_upper, upper_bounds, steps);
            return indexing.dorder_index_iterator(...bounds);
        }

        /**
         * Create an iterator over the values of the array, in data order.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         * @private
         */
        _dorder_value_iterator(lower_or_upper?: Uint32Array, upper_bounds?: Uint32Array, steps?: Uint32Array): Iterable<number> {

            const index_iterator = this._dorder_data_iterator(lower_or_upper, upper_bounds, steps);
            const self = this;
            const iter = {
                [Symbol.iterator]: function* () {
                    for (let index of index_iterator) {
                        yield self.data[index];
                    }
                }
            }

            return iter;
        }


        /**
         * Compute the lower bounds, upper bounds, and steps for a slice.
         * @param lower_or_upper - The lower bounds of the slice if upper_bounds is defined. Otherwise this is the upper_bounds, and the lower bounds are the offset of the tensor.
         * @param upper_bounds - The upper bounds of the slice. Defaults to the shape of the tensor.
         * @param steps - The size of the steps to take along each axis.
         */
        private _calculate_slice_bounds(lower_or_upper: Uint32Array, upper_bounds: Uint32Array, steps: Uint32Array): [Uint32Array, Uint32Array, Uint32Array] {
            let lower_bounds;
            if (lower_or_upper === undefined) {
                lower_bounds = new Uint32Array(this.shape.length);
                upper_bounds = this.shape;
            } else if (upper_bounds === undefined) {
                lower_bounds = new Uint32Array(this.shape.length);
                upper_bounds = lower_or_upper;
            } else {
                lower_bounds = lower_or_upper;
            }

            if (steps === undefined) {
                steps = utils.fixed_ones(this.shape.length);
            }

            return [lower_bounds, upper_bounds, steps];
        }

    //#endregion Indexing

    //#region Binary Methods

        /**
         * Add `b` to `this`.
         * @param b - The value to add to the array.
         */
        add(b: Broadcastable): tensor {
            return arithmetic._add(this, b);
        }

        /**
         * Subtract a broadcastable value from this.
         * @param {Broadcastable} b - Value to subtract.
         * @return {number | tensor}
         */
        sub(b: Broadcastable): tensor {
            return arithmetic._sub(this, b);
        }

        /**
         * Multiply `this` by `b`.
         * @param b - A tensor to multiply by.
         */
        mult(b: Broadcastable): tensor {
            return arithmetic._mult(this, b);
        }

        /**
         * Divide `this` by `b`.
         * @param b - A tensor to divide by.
         */
        div(b: Broadcastable): tensor {
            return arithmetic._div(this, b);
        }

        mod(b: Broadcastable): tensor {
            return arithmetic._mod(this, b);
        }

        /**
         * Return true if this array equals the passed array, false otherwise.
         * @param {tensor} a  - The array to compare against.
         * @return {boolean}
         */
        equals(a: tensor): boolean {
            return tensor.equals(this, a);
        }

        /**
         *  Return an array of booleans. Each entry is whether the corresponding entries in a and b are numerically close. The arrays will be broadcasted. 
         * @param b - Second array to compare. 
         * @param rel_tol - The maximum relative error.
         * @param abs_tol - The maximum absolute error.
         */
        is_close(b: tensor, rel_tol: number = 1e-5, abs_tol: number = 1e-8): tensor {
            const compare = (x: number, y: number): number => {
                return +(Math.abs(x - y) <= abs_tol + (rel_tol * Math.abs(y)));
            }
            return tensor._binary_broadcast(this, b, compare);
        }

    //#endregion Binary Methods

    //#region OPERATIONS

        /**
         * Convert a broadcastable value to a tensor.
         * @param {Broadcastable} value - The value to convert. Numbers will be converted to 1x1 tensors, TypedArrays will be 1xn, and tensors will be left alone.
         * @return {tensor}           - The resulting tensor.
         * @private
         */
        private static _upcast_to_tensor(value: Broadcastable): tensor {
            let a_array;
            if (utils.is_numeric(value)) {
                a_array = tensor.array(new Float64Array([value]), new Uint32Array([1]), { disable_checks: true });
            } else if (utils.is_typed_array(value)) {
                a_array = tensor.array(value, new Uint32Array([value.length]), { disable_checks: true });
            } else {
                a_array = value;
            }
            return a_array;
        }

        /**
         * Broadcast two values together.
         * Works like numpy broadcasting.
         * @param {Broadcastable} a - The first broadcastable value.
         * @param {Broadcastable} b - The second broadcastable value.
         * @return {[IterableIterator<number[]>, Uint32Array, string]}  - An iterator over that returns a tuple (a_i, b_i) of broadcasted values, the new shape, and the new dtype.
         * @private
         */
        private static _broadcast_by_index(a: Broadcastable, b: Broadcastable): [IterableIterator<[number, number, Uint32Array]>, Uint32Array, string] {

            let a_array = tensor._upcast_to_tensor(a);
            let b_array = tensor._upcast_to_tensor(b);

            const new_dimensions = indexing.calculate_broadcast_dimensions(a_array.shape, b_array.shape);
            const new_dtype = utils._dtype_join(a_array.dtype, b_array.dtype);
            let index_iter = indexing.iorder_index_iterator(new_dimensions);

            const iterator = utils.zip_longest(a_array._iorder_data_iterator(), b_array._iorder_data_iterator(), index_iter);

            let iter = {};
            iter[Symbol.iterator] = function* () {
                for (let [a_index, b_index, index] of iterator) {
                    const a_val = a_array.data[a_index];
                    const b_val = b_array.data[b_index];
                    yield [a_val, b_val, index];
                }
            };

            return [<IterableIterator<[number, number, Uint32Array]>>iter, new_dimensions, new_dtype];
        }

        /**
         * Apply a binary function to two broadcastables.
         * @param {Broadcastable} a - The first argument to f.
         * @param {Broadcastable} b - The second argument to f.
         * @param {(a: number, b: number) => number} f  - The function to apply.
         * @param {string} dtype  - Optional forced data type.
         * @return {tensor}  - The result of applying f to a and b.
         * @private
         */
        static _binary_broadcast(a: Broadcastable, b: Broadcastable, f: (a: number, b: number) => number, dtype?: string): tensor {
            let [iter, shape, new_dtype] = tensor._broadcast_by_index(a, b);

            if (dtype === undefined) {
                dtype = new_dtype
            }

            let new_array = tensor.filled(0, shape, dtype);

            for (let [a_val, b_val, index] of iter) {
                const new_val = f(a_val, b_val);
                new_array.s(new_val, ...index);
            }

            return new_array
        }

        /**
         * 
         * @param {Broadcastable} a -
         * @param {Broadcastable} b -
         * @returns {tensor}      -
         */
        static broadcast_matmul(a: Broadcastable, b: Broadcastable): tensor {
            let a_array = tensor._upcast_to_tensor(a);
            let b_array = tensor._upcast_to_tensor(b);

            const a_shape: Uint32Array = a_array.shape;
            const b_shape: Uint32Array = b_array.shape;

            // Check they can actually be multiplied.
            if (a_shape[a_shape.length - 1] !== b_shape[b_shape.length - 2]) {
                throw new Error(`Shapes ${a_shape} and ${b_shape} are not aligned for matrix multiplication.`);
            }

            const broadcast = indexing.calculate_broadcast_dimensions(a_array.shape.slice(0, -2), b_array.shape.slice(0, -2));
            const new_dimensions = new Uint32Array([...broadcast,
            a_shape[a_shape.length - 2],
            b_shape[b_shape.length - 1]
            ]);

            if (new_dimensions.length === 2) {
                return tensor.matmul_2d(a_array, b_array);
            } else {
                const new_dtype = utils._dtype_join(a_array.dtype, b_array.dtype);
                let array = tensor.zeros(new_dimensions, new_dtype);

                const index_iter = indexing.iorder_index_iterator(new_dimensions.slice(0, -2));
                const a_iter = indexing.iorder_index_iterator(a_shape.slice(0, -2));
                const b_iter = indexing.iorder_index_iterator(b_shape.slice(0, -2));
                const iter = utils.zip_longest(a_iter, b_iter, index_iter);
                for (let [a_index, b_index, index] of iter) {
                    const slice = indexing.index_to_slice(index);

                    const b1 = b_array.slice(...b_index);
                    const a1 = a_array.slice(...a_index);
                    const subarray = tensor.matmul_2d(a1, b1);

                    array.s(subarray, ...slice);
                }
                return array;
            }
        }

        /**
         * Multiply two 2D matrices.
         * Computes a x b.
         * @param {tensor} a  - The first array. Must be m x n.
         * @param {tensor} b  - The second array. Must be n x p.
         * @returns {tensor}  - The matrix product.
         */
        static matmul_2d(a: tensor, b: tensor): tensor {
            const new_shape = new Uint32Array([a.shape[0], b.shape[1]]);

            let iter = {
                [Symbol.iterator]: function* () {
                    for (let i = 0; i < new_shape[0]; i++) {
                        for (let j = 0; j < new_shape[1]; j++) {
                            const a_vec = a.slice(i);
                            const b_vec = b.slice(null, j);
                            let x = tensor.dot(a_vec, b_vec);
                            yield x;
                        }
                    }
                }
            };

            return tensor.from_iterable(iter, new_shape);
        }

        // TODO: Generalize to an inner product.
        // TODO: This is numerically unstable.
        /**
         * Compute the dot product of two arrays.
         * @param {tensor} a
         * @param {tensor} b
         * @return {number}
         */
        static dot(a: tensor, b: tensor): number {
            let acc = 0;
            let a_iter = a._iorder_value_iterator();
            let b_iter = b._iorder_value_iterator();
            for (let [a_val, b_val] of utils.zip_iterable(a_iter[Symbol.iterator](), b_iter[Symbol.iterator]())) {
                acc += a_val * b_val;
            }
            return acc;
        }

        /**
         * Create an array containing the element-wise max of the inputs.
         * Inputs must be the same shape.
         * @param {tensor} a  - First array.
         * @param {tensor} b  - Second array.
         * @return {tensor}   - An array with the same shape as a and b. Its entries are the max of the corresponding entries of a and b.
         */
        static take_max(a: tensor, b: tensor) {
            return tensor._binary_broadcast(a, b, (x, y) => Math.max(x, y));
        }

        /**
         * Create an array containing the element-wise min of the inputs.
         * Inputs must be the same shape.
         * @param {tensor} a  - First array.
         * @param {tensor} b  - Second array.
         * @return {tensor}   - An array with the same shape as a and b. Its entries are the min of the corresponding entries of a and b.
         */
        static take_min(a: tensor, b: tensor) {
            return tensor._binary_broadcast(a, b, (x, y) => Math.min(x, y));
        }

        /**
         * Compute the element-wise quotient of two arrays, rounding values up to the nearest integer.
         * @param {Broadcastable} a - Dividend array.
         * @param {Broadcastable} b - Divisor array.
         * @return {Broadcastable}  - Quotient array.
         */
        static _cdiv(a: Broadcastable, b: Broadcastable): tensor {
            return tensor._binary_broadcast(a, b, (x, y) => Math.ceil(x / y));
        }

        /**
         * Compute the element-wise quotient of two arrays, rounding values down to the nearest integer.
         * @param {Broadcastable} a - Dividend array.
         * @param {Broadcastable} b - Divisor array.
         * @return {tensor}       - Quotient array.
         */
        static _fdiv(a: Broadcastable, b: Broadcastable): tensor {
            return tensor._binary_broadcast(a, b, (x, y) => Math.floor(x / y));
        }

    //#endregion OPERATIONS

    /**
     * Check if two n-dimensional arrays are equal.
     * @param {tensor} array1
     * @param {tensor} array2
     * @return {boolean}
     */
    static equals(array1: tensor, array2: tensor): boolean {
        return (
            (array1.length === array2.length) &&
            (tensor._equal_data(array1.shape, array2.shape)) &&
            (tensor._equal_data(array1.offset, array2.offset)) &&
            (tensor._equal_data(array1.stride, array2.stride)) &&
            (tensor._equal_data(array1.dstride, array2.dstride)) &&
            (array1.initial_offset === array2.initial_offset) &&
            (array1.dtype === array2.dtype) &&
            (tensor._equal_data(array1, array2))
        );
    }

    /**
     * Check if two arraylikes have the same length and the same elements.
     * @param array1
     * @param array2
     * @return {boolean}  - true if the length and elements match, false otherwise.
     * @private
     */
    static _equal_data(array1, array2): boolean {
        if (array1 instanceof tensor) {
            array1 = array1.data;
        }

        if (array2 instanceof tensor) {
            array2 = array2.data;
        }

        return (
            (array1.length === array2.length) &&
            (array1.reduce((a, e, i) => a && e === array2[i], true))
        );
    }

    /**
     * Return a copy of a.
     * @param {tensor} a  - tensor to copy.
     * @return {tensor}   - The copy.
     */
    static copy(a: tensor): tensor {
        const new_data = a.data.slice(0);
        return new tensor(new_data, a.shape.slice(0), a.offset.slice(0), a.stride.slice(0), a.dstride.slice(0), a.length, a.dtype, a.is_view, a.initial_offset);
    }

    //#region CONSTRUCTORS

        /**
         * Convert the tensor to a nested JS array.
         */
        to_nested_array(): Array<any> {
            let array = [];
            for (let index of this._iorder_index_iterator()) {
                let subarray = array;
                for (let i of index.slice(0, -1)) {
                    if (subarray[i] === undefined) {
                        subarray[i] = [];
                    }
                    subarray = subarray[i];
                }
                subarray[index[index.length - 1]] = this.g(...index);
            }
            return array;
        }

        /**
         * Convert the tensor to JSON.
         */
        to_json(): object {
            let json = {
                data: this.to_nested_array(),
                shape: Array.from(this.shape),
                dtype: this.dtype
            };
            return json;
        }

        /**
         * Create a tensor from JSON.
         * @param json - The JSON representation of the array.
         */
        static from_json(json: any): tensor {
            return tensor.from_nested_array(json.data, json.dtype);
        }

        /**
         * Create a tensor from a nested array of values.
         * @param {any[]} array - An array of arrays (nested to arbitrary depth). Each level must have the same dimension.
         * The final level must contain valid data for a tensor.
         * @param {string} dtype  - The type to use for the underlying array.
         *
         * @return {tensor}
         */
        static from_nested_array(array: any[], dtype?: string): tensor {
            if (array.length === 0) {
                return tensor.array([]);
            }

            const dimensions = utils._nested_array_shape(array);
            let slice_iter = indexing.iorder_index_iterator(dimensions);

            const size = indexing.compute_size(dimensions);
            const array_type = utils.dtype_map(dtype);
            const data = new array_type(size);

            let ndarray = tensor.array(data, dimensions, { dtype: dtype, disable_checks: true });

            for (let indices of slice_iter) {
                const real_index = ndarray._compute_real_index(indices);
                ndarray.data[real_index] = utils._nested_array_value_from_index(array, indices);
            }

            return ndarray;
        }

        /**
         * Create an n-dimensional array from an iterable.
         * @param iterable
         * @param shape
         * @param {string} dtype
         * @return {tensor}
         */
        static from_iterable(iterable: Iterable<number>, shape: Shape, dtype?: string) {
            const final_shape = indexing.compute_shape(shape);

            const size = indexing.compute_size(final_shape);
            const array_type = utils.dtype_map(dtype);
            const index_iterator = indexing.iorder_index_iterator(final_shape);
            const val_gen = iterable[Symbol.iterator]();
            let data = new array_type(size);
            const stride = indexing.stride_from_shape(final_shape);
            const initial_offset = 0;
            let i = 0;
            for (let index of index_iterator) {
                const real_index = indexing.index_in_data(index, stride, initial_offset);
                let val = val_gen.next();
                data[real_index] = val.value;
            }

            if (data.length !== size) {
                throw new errors.MismatchedShapeSize(`Iterable passed has size ${data.length}. Size expected from shape was: ${size}`);
            }

            return tensor.array(data, final_shape, { disable_checks: true, dtype: dtype });
        }

        /**
         * Produces an array of the desired shape filled with a single value.
         * @param {number} value                - The value to fill in.
         * @param shape - A numerical array or a number. If this is a number a one-dimensional array of that length is produced.
         * @param {string} dtype                - The data type to use for the array. float64 by default.
         * @return {tensor}
         */
        static filled(value: number, shape, dtype?: string): tensor {
            const final_shape = indexing.compute_shape(shape);

            const size = indexing.compute_size(final_shape);
            const array_type = utils.dtype_map(dtype);
            const data = new array_type(size).fill(value);

            return tensor.array(data, final_shape, { disable_checks: true, dtype: dtype });
        }

        /**
         * Return an array of the specified size filled with zeroes.
         * Equivalent to `tensor.filled`, but slightly faster.
         * @param {number} shape
         * @param {string} dtype
         * @return {tensor}
         */
        static zeros(shape, dtype?: string) {
            const final_shape = indexing.compute_shape(shape);
            const size = indexing.compute_size(final_shape);
            const array_type = utils.dtype_map(dtype);
            const data = new array_type(size);

            return tensor.array(data, final_shape, { disable_checks: true, dtype: dtype });
        }

        /**
         * Create an identity matrix of a given size.
         * @param m - The size of the identity matrix.
         * @param dtype - The dtype for the identity matrix.
         */
        static eye(m: number, dtype?: string): tensor {
            let array = tensor.zeros([m, m], dtype);
            for (let i = 0; i < m; i++) {
                array.s(1, i, i);
            }
            return array;
        }

        /**
         * Create a tensor containing the specified data
         * @param data
         * @param shape
         * @param options
         * @return {tensor}
         */
        static array(data, shape?, options?: ArrayOptions): tensor {
            let final_shape;
            let size;
            let dtype;

            if (shape === undefined) {
                shape = new Uint32Array([data.length]);
            }

            if (options && options.dtype) {
                dtype = options.dtype
            }

            if (options && options.disable_checks === true) {
                final_shape = shape;
                size = indexing.compute_size(shape);
            } else {
                if (!utils.is_numeric_array(data)) {
                    throw new errors.BadData();
                }

                if (shape === undefined || shape === null) {
                    final_shape = new Uint32Array([data.length]);
                } else {
                    final_shape = indexing.compute_shape(shape);
                }

                // Compute length
                size = indexing.compute_size(final_shape);

                if (size !== data.length) {
                    throw new errors.MismatchedShapeSize()
                }
            }

            const stride = indexing.stride_from_shape(final_shape);
            const offset = new Uint32Array(final_shape.length);
            const dstride = stride.slice();

            return new tensor(data, final_shape, offset, stride, dstride, size, dtype);
        }

    //#endregion CONSTRUCTORS

}

export { errors };
