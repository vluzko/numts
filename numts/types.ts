import {tensor} from "./tensor";

export type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array| Int32Array | Uint32Array | Float32Array | Float64Array;
export type Numeric = TypedArray | number[];
export type Broadcastable = number | TypedArray | tensor | number[];
export type Shape = number[] | Uint32Array;
export type USlice = Array<null | number | [number, number] | [number, number, number]>;
export type ISlice = Array<[number, number, number]>;
