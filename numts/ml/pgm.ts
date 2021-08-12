/** Probabilistic graphical models */

import {tensor} from '../tensor';

/**
 * A simple DAG implementation
 */
class DAG {
    public nodes: Uint32Array;
    public adjacency_list: Map<number, [number[], number[]]>;

    constructor(nodes: Uint32Array, adjacency_list: Map<number, [number[], number[]]>) {
        this.nodes = nodes;
        this.adjacency_list = adjacency_list;
    }
}


export class BayesNet {
    public dag: DAG;

    /**
     * Find the conditional probability distributions that best fit
     * @param data - The data to fit.
     */
    fit(data: tensor) {

    }

    /**
     * Predict the values of the unknown nodes
     * @param input - The known nodes
     */
    predict(input: Map<number, number>) {

    }
}
