function random_shape(max_size, upper) {
    if (max_size === undefined) {
      max_size = 10;
    }

    if (upper === undefined) {
      upper = 10;
    }

    let shape = [];
    for (let i = 0; i < max_size; i++) {
      shape.push(Math.ceil(Math.random() * upper))
    }

    return new Uint32Array(shape);
}


module.exports = {random_shape};
