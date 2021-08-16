import { tensor } from "./tensor";
import { zeros, eye } from "./numts";

//#region utils
/**
 * 
 * @param a - The array.
 */
export function column_iterator(a: tensor) {
    const step = a.shape[0];
    let iter = {
        [Symbol.iterator]: function* () {
            let index = 0;
            for (let column = 0; column < a.shape[1]; column++) {
                let iter2 = {
                    [Symbol.iterator]: function* () {
                        for (let row = 0; row < a.shape[0]; row++) {
                            yield a.data[index];
                            index += 1;
                        }
                    }
                }
                yield iter2;
            }
        }
    }
    return iter;
}

/**
 * Return true if the tensor is one-dimensional.
 */
export function is_vector(a: tensor): boolean {
    return a.shape.length === 1;
}

export function is_flat(a: tensor): boolean {
    return (a.shape.length === 1) || (a.shape.reduce((x, y) => y === 1 ? x : x + 1, 0) <= 1);
}

/**
 * Return true if the tensor is two-dimensional.
 */
export function is_matrix(a: tensor): boolean {
    return a.shape.length === 2;
}

/**
 * Return true if the tensor is row or column vector.
 */
export function is_flat_matrix(a: tensor): boolean {
    if (a.shape.length === 2) {
        const [m, n] = a.shape;
        return (m === 1) || (n === 1);
    } else {
        return false;
    }
}

/**
 * Return true if the tensor is a square matrix.
 */
export function is_square(a: tensor): boolean {
    return a.shape.length === 2 && a.shape[0] === a.shape[1];
}

export function is_diagonal(a: tensor): boolean {
    throw new Error('Not implemented')
}

export function is_bidiagonal(a: tensor): boolean {
    throw new Error('Not implemented.')
}

//#endregion utils

/**
 * Calculate the L1 norm of a tensor.
 * @param a - A tensor.
 */
export function l1(a: tensor) {
    if (is_vector(a)) {
        return a.reduce((x, y) => x + Math.abs(y), 0);
    } else if (is_matrix(a)) {
        let max = Number.MIN_VALUE;
        for (let column of column_iterator(a)) {
            let sum = 0;
            for (let value of column) {
                sum += Math.abs(value);
            }
            max = sum > max ? sum : max;
        }
        return max;
    }
}

/**
 * Calculate the L2 norm of the tensor.
 * @param a - A tensor.
 */
export function l2(a: tensor): number {
    // Calculate sigma_1 of a.
    if (is_flat(a)) {
        // @ts-ignore
        return Math.sqrt(a.reduce((x, y) => x + Math.pow(y, 2), 0));
    } else {
        throw new Error('Not implemented');
    }
}

/**
 * Calculate the L_infinity norm of a tensor.
 * @param a - A tensor
 */
export function linf(a: tensor) {

}

/**
 * Calculate the Frobenius norm of a tensor.
 * @param a - A tensor.
 */
export function fnorm(a: tensor) {

}

export function pnorm(a: tensor) {

}

export function inv(a: tensor): tensor {
    throw new Error();
}

//#region Decompositions
/**
 * Calculate the singular valud decomposition of a matrix.
 * @param a - A matrix.
 */
export function svd(a: tensor): [tensor, tensor, tensor] {
    if (!is_matrix(a)) {
        throw new Error(`Can only calculate SVD for square matrices.`);
    } 
    const m = a.shape[0];
    const n = a.shape[1];

    // Express A as U_1 x B x V_1
    let [u, s, v] = householder_bidiagonal(a);

    const epsilon = 0.00001
    let done = false;
    while (!done) {
        // Zero out small superdiagonal entries

        // Sweep
    }
    // Use Givens rotations to diagonalize s.
    // let givens: tensor;
    // for (let i = 0; i < m-1; i++) {
    //     console.log(`Row: ${i}`);
    //     // if (i == 2) debugger;
    //     // Skip this row if it's already zeroed
    //     if (s.g(i, i+1) == 0) {
    //         continue
    //     } else {
    //         // Eliminate right of diagonal
    //         [givens, s] = givens_rotation_row(s, i, i+1);
    //         v = tensor.matmul_2d(givens, v);

    //         // Eliminate new entry under diagonal
    //         [givens, s] = givens_rotation_up(s, i, i+1);
    //         // console.log(s.g(i+1, i))

    //         u = tensor.matmul_2d(u, givens);
    //     }
    // }

    // U_1 U_k and V_k V_1 are U and V respectively
    return [u, s, v]
}

/**
 * Sweep the superdiagonal of a matrix with Givens rotations.
 */
export function svd_sweep(b: tensor): [tensor, tensor, tensor] {
    // Givens rotation of first superdiagonal
    // Givens rotation of first subdiagonal
    // Iterate
    throw new Error('Not implemented')
}


/**
 * LU decomposition of a square matrix.
 * Uses the Doolittle algorithm.
 * @param a - The matrix to factor.
 * 
 * @returns - A tuple containing the L and U matrices.
 */
export function lu(a: tensor): [tensor, tensor] {
    if (!is_square(a)) {
        throw new Error('LU decomposition is only valid for square matrices.');
    }
    const n = a.shape[0];
    let lower = zeros(a.shape);
    let upper = zeros(a.shape);
    for (let i = 0; i < n; i++) {
        for (let k = i; k < n; k++) {
            const sum = tensor.dot(lower.slice(i), upper.slice(null, k));
            const diff = a.g(i, k) - sum;
            upper.s(diff, i, k);
        }

        for (let k = i; k < n; k++) {
            if (i === k) {
                lower.s(1, i, i);
            } else {
                const sum = tensor.dot(lower.slice(k), upper.slice(null, i));
                const diff = (a.g(k, i) - sum) / upper.g(i, i);

                lower.s(diff, k, i);
            }
        }
    }

    return [lower, upper];
}

/**
 * Calculate the Cholesky decomposition of a matrix.
 * @param a - A matrix.
 */
export function chol(a: tensor) {

}

export function rank(a: tensor) {

}

/**
 * Calculate the QR decomposition of a matrix.
 * @param a - A matrix.
 * @param options - Options.
 */
export function qr(a: tensor, options = { just_r: false, algorithm: 'givens' }) {

    if (!is_matrix(a)) {
        throw new Error('QR only works for matrices');
    }

    const just_r = options.just_r === undefined ? false : options.just_r;
    const algorithm = options.algorithm === undefined ? 'givens' : options.algorithm;

    if (algorithm === 'givens') {
        return givens_qr(a);
    } else if (algorithm === 'householder') {
        return householder_qr(a);
    } else {
        throw new Error(`Unknown algorithm "${algorithm}"`);
    }
}

/**
 * QR factorization of a symmetric matrix.
 * @param a - The matrix to decompose.
 */
export function sym_qr(a: tensor): [tensor, tensor] {
    throw new Error();
}

//#region Householder QR
/**
 * Calculate a QR decomposition of `a` using Householder transformations.
 * @param A - The matrix to decompose.
 */
function householder_qr(A: tensor) {
    const [m, n] = A.shape;
    let Q = eye(m);
    let R = A.as_type('float64');
    if (m === 1 && n === 1) {
        return [Q, R];
    }
    for (let j = 0; j < n; j++) {
        // Calculate the vector to reflect around.
        const lower_column = R.slice([j, null], [j, j + 1]);
        // @ts-ignore
        const norm = l2(lower_column);
        // If the norm is already very close to zero, the column is already
        if (norm < 1e-14) {
            continue
        } else {
            const pivot: number = R.g(j, j);
            const s: number = pivot >= 0 ? 1 : -1;
            const u1: number = pivot + s * norm;
            const normalized: tensor = lower_column.div(u1);
            normalized.s(1, 0);
            const tau: number = s * u1 / norm;
            const tauw = normalized.mult(tau);

            // Update R
            const r_block = R.slice([j, null], null);
            const temp1 = tensor.matmul_2d(normalized.transpose(), r_block);
            const temp2 = tensor.matmul_2d(tauw, temp1);
            const r_diff = r_block.sub(temp2);
            R.s(r_diff, [j, null], null);

            // Update Q
            const q_block = Q.slice(null, [j, null]);
            const matmul = tensor.matmul_2d(q_block, normalized);
            const temp3 = tensor.matmul_2d(matmul, tauw.transpose());
            const q_diff = q_block.sub(temp3);
            Q.s(q_diff, null, [j, null]);
        }

    }
    return [Q, R];
}

/**
 * Calculate the vector required to create a Householder transform canceling the given column
 * @param a - The matrix to transform. Transformation is done in place.
 * @param i - The row index of the pivot.
 * @param j - The column index of the pivot.
 */
export function householder_col_vector(a: tensor, i: number, j: number): [tensor, number] {
    let vec = tensor.copy(a.slice([i, null], [j, j + 1]));
    const pivot: number = a.g(i, j);
    return compute_householder(vec, pivot);
}

/**
 * Calculate the vector required to create a Householder transform canceling the given row.
 * @param a - The matrix to transform. Transformation is done in place.
 * @param i - The row index of the pivot.
 * @param j - The column index of the pivot.
 */
export function householder_row_vector(a: tensor, i: number, j: number): [tensor, number] {
    let vec = tensor.copy(a.slice([i, i + 1], [j, null]));
    const pivot: number = a.g(i, j);
    return compute_householder(vec, pivot);
}

/**
 * Given a row/column vector and the value of the pivot, compute the required Householder vector.
 * Note that only the *non-zero* entries of the w vector are returned, since these
 * are all that is required to calculate the Householder matrix.
 * Algorithm 5.1.1 in Golub & van Loan, 4th Edition.
 * TODO: Optimization: Pass sigma in directly so we don't need the first component of the vector
 * (which always ends up equal to 1)
 */
function compute_householder(vec: tensor, pivot: number): [tensor, number] {
    const sigma = <number>vec.reduce((x, y) => x + Math.pow(y, 2), 0) - Math.pow(pivot, 2);
    let beta: number;
    // If the norm is already very close to zero, the column is already zeroed.
    if (sigma < 1e-14) {
        beta = 0.0;
    } else {
        const mu = Math.sqrt(Math.pow(pivot, 2) + sigma);
        let val: number;
        if (pivot <= 0) {
            val = pivot - mu;
        } else {
            val = -sigma / (pivot + mu);
        }
        const sval = Math.pow(val, 2);
        vec.s(val, 0, 0);
        beta = 2 * sval / (sigma + sval)
        vec = vec.div(val);
    }

    return [vec, beta];
}

/**
 * Calculate the full Householder transformation for a column.
 */
export function full_h_col_matrix(w: tensor, m: number, beta: number): tensor {
    let h = eye(m);
    const [i, _] = w.shape;
    const squared = tensor.matmul_2d(w, w.transpose());
    const w_beta = squared.mult(beta);
    const index = m - i;

    const diff = h.slice([index, null], [index, null]).sub(w_beta);
    h.s(diff, [index, null], [index, null]);
    return h;
}

/**
 * Calculate the full Householder transformation for a row
 */
export function full_h_row_matrix(w: tensor, m: number, beta: number): tensor {
    let h = eye(m);
    const [_, j] = w.shape;
    const squared = tensor.matmul_2d(w.transpose(), w);
    const w_beta = squared.mult(beta);
    const index = m - j;

    const diff = h.slice([index, null], [index, null]).sub(w_beta);
    h.s(diff, [index, null], [index, null]);

    return h;

}

/**
 * Reduce an [m, n] matrix to bidiagonal form using Householder transformations.
 * This is Golub-Kahan bidiagonalization.
 * TODO: Optimization: Use the zeroed entries of the matrix to store the Householder transforms
 * TODO: Optimization: Run Lawson-Hanson-Chan insteadif m > 5/3n
 * (as suggested by GVL)
 */
export function householder_bidiagonal(original: tensor): [tensor, tensor, tensor] {
    let a = tensor.copy(original);
    const [m, n] = a.shape;
    if (m < n) {throw new Error(`SVD needs a tall triangular matrix. Got (${m}, ${n})`);}

    let u = eye(m);
    let v = eye(n);

    for (let col = 0; col < n; col++) {

        // Zero out the column.
        const [w, beta] = householder_col_vector(a, col, col);
        // TODO: Optimization: This can be optimized by just iterating over the transpose.
        // TODO: Optimization: The first row and column of this difference are always
        // just copies of w, since w[0] = 1.0
        const w_square = tensor.matmul_2d(w, w.transpose()).mult(beta);
        // Householder matrix to zero out col.
        // TODO: Optimization: Compute the Householder matrix as it's being created.
        let householder_matrix = eye(m - col);
        const diff = householder_matrix.sub(w_square);
        householder_matrix.s(diff, [0, null], [0, null])
        const a_col_slice = a.slice([col, null], [col, null]);
        const lower_slice = tensor.matmul_2d(householder_matrix, a_col_slice);
        a.s(lower_slice, [col, null], [col, null]);

        // Update U
        // TODO: Optimization: The fact that the first (col - 1) rows/columns of the
        // Householder matrix are just an identity matrix means this can be optimized.
        // The first (col - 1) columns of u will not be updated by this.
        // (This may change if doing fat matrices, I'm not sure)
        let full_house_m = eye(m);
        full_house_m.s(householder_matrix, [col, null], [col, null]);
        u = tensor.matmul_2d(u, full_house_m);

        // Zero out the row.
        if (col <= n - 2) {
            const [w, beta] = householder_row_vector(a, col, col+1);
            const w_square = tensor.matmul_2d(w.transpose(), w).mult(beta);
            let householder_matrix = eye(n - col);
            const householder_slice = householder_matrix.slice([1, null], [1, null]);
            householder_matrix.s(householder_slice.sub(w_square), [1, null], [1, null])
            const a_col_slice = a.slice([col, null], [col, null]);
            const lower_slice = tensor.matmul_2d(a_col_slice, householder_matrix);
            a.s(lower_slice, [col, null], [col, null]);

            // Update V
            const v_update = tensor.matmul_2d(v.slice([col, null], [col, null]), householder_matrix);
            v.s(v_update, [col, null], [col, null]);
        }
    }

    return [u, a, v];
}

function pivoted_householder(a: tensor): [tensor, Uint32Array] {
    throw new Error();
    // if (!is_matrix(a)) {
    //     throw new Error()
    // }

    // const [m, n] = a.shape;
    // let c = [];
    // // Stores the permutation matrix.
    // // The final permutation matrix is the identity matrix with row j and row pivot[j] swapped.
    // let pivot = new Uint32Array(m);
    // let tau = Number.MIN_VALUE;
    // let k;
    // for (let i = 0; i < n; i++) {
    //     const l = a.slice(null, i);
    //     c[i] = l.dot(l)
    //     if (c[i] >= tau) {
    //         tau = c[i];
    //         k = i;
    //     }
    // }
    // let r = 0;
    // while (tau > 0 && r < n) {
    //     pivot[r] = k;
    //     // Swap the chosen columns
    //     const col_r = a.slice(null, r);
    //     // Set column r to column k
    //     a.s(a.slice(null, k), null, r)
    //     // Set column k to column
    //     a.s(col_r, null, k);

    //     const c_k = c[k];
    //     c[k] = c[r];
    //     c[r] = c_k;

    //     const [v, b] = householder_vector(a.slice([r, null], r));
    //     const h = construct_householder_matrix(v, b);
    //     const vals = tensor.matmul_2d(h, a.slice([r, null], [r, null]));
    //     a.s(vals, [r, null], [r, null]);
    //     a.s(v.slice([1, m - r + 1]), [r+1, null], r);
    //     tau = Number.MIN_VALUE;
    //     for (let j = r+1 ; j < n; j++) {
    //         c[j] = c[j] - Math.pow(a.g(r, j), 2);
    //         if (c[j] > tau) {
    //             tau = c[j];
    //             k = j;
    //         }
    //     }
    // }
    // return [a, pivot];
}

//#endregion Householder QR

//#region Givens QR
/**
 * Calculate the QR factorization of `a` using Givens rotations.
 * @param A - The matrix to factorize.
 */
function givens_qr(A: tensor): [tensor, tensor] {
    const [m, n] = A.shape;
    let Q = null;
    let R = A;
    // Zero out the lower diagonal.
    for (let i = 0; i < n; i++) {
        for (let j = m - 1; j > i; j--) {
            let G;
            [G, R] = givens_rotation_up(R, i, j);
            if (Q === null) {
                Q = G;
            } else {
                Q = tensor.matmul_2d(G, Q);
            }
        }
    }

    // Handle one-dimensional arrays.
    if (Q === null) {
        Q = eye(m);
    }

    return [Q.transpose(), R];
}

/**
 * 
 * @param A - The matrix to perform the rotation on.
 * @param i - The row to rotate to.
 * @param j - The row to rotate from, and the column.
 */
export function givens_rotation_up(A: tensor, i: number, j: number): [tensor, tensor] {
    // debugger;
    const bottom_val = A.g(j, i);
    const top_val = A.g(i, i);
    // const r = Math.sqrt(Math.pow(bottom_val, 2) + Math.pow(top_val, 2));
    // const s = bottom_val / r;
    // const c = top_val / r;
    const [c, s] = givens_values(top_val, bottom_val);
    const [m, n] = A.shape;
    let G = eye(m);
    G.s(c, i, i);
    G.s(c, j, j);
    G.s(-s, i, j);
    G.s(s, j, i);
    // console.log(G.to_nested_array())
    const R = tensor.matmul_2d(G, A);
    console.log(R.to_nested_array());
    return [G, R];
}

/**
 * Use Givens rotation to zero out a row entry against its diagonal.
 * Based on Algorithm 5.1.3 in Golub & van Loan 4th edition.
 * @param A - The matrix to perform the rotation on.
 * @param i - The row to rotate to.
 * @param j - The row to rotate from, and the column.
 */
export function givens_rotation_row(A: tensor, i: number, j: number): [tensor, tensor] {
    const [m, ] = A.shape;
    // a in G & vL
    const diagonal = A.g(i, i);
    // b in G & vL
    const right_val = A.g(i, j);
    let G = eye(m);
    let R: tensor;
    if (right_val == 0) {
        R = tensor.copy(A);
    } else {
        const [c, s] = givens_values(diagonal, right_val);

        G.s(c, i, i);
        G.s(c, j, j);
        G.s(s, i, j);
        G.s(-s, j, i);

        R = tensor.matmul_2d(A, G);
    }
    console.log(R.to_nested_array());
    return [G, R];

    // const r = Math.sqrt(Math.pow(right_val, 2) + Math.pow(diagonal, 2));
    // const s = right_val / r;
    // const c = diagonal / r;
    // let G = eye(m);


    // // TODO: Optimization: This can be much more efficient, most of the multiplications can be skipped.
    // console.log(R.to_nested_array())
    // return [G, R];
}

/**
 * Calculate c and s for a Givens rotation.
 * c is on diagonal, s is off diagonal (cosine and sine).
 */
export function givens_values(a: number, b: number): [number, number] {
    let s: number;
    let c: number
    if (b == 0) {
        c = 1;
        s = 0;
    } else {
        if (Math.abs(b) > Math.abs(a)) {
            const tau = -a / b;
            s = 1 / Math.sqrt(1 + Math.pow(tau, 2));
            c = s * tau;
        } else {
            const tau = -b / a;
            c = 1 / Math.sqrt(1 + Math.pow(tau, 2));
            s = c * tau;
        }
    }

    return [c, s];
}

/**
 * Multiply a compressed Givens rotation matrix by a normal matrix.
 * @param G - The compressed representation of the Givens rotation.
 * @param A - The matrix being transformed.
 */
function compressed_givens_mult(G: Float64Array, A: tensor): tensor {
    throw new Error();
}

//#endregion Givens QR

//#endregion Decompositions
