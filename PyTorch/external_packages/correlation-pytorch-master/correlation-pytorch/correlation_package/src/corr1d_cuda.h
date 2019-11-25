int corr1d_cuda_forward(
    torch::Tensor &input1,
    torch::Tensor &input2,
    torch::Tensor &rbot1,
    torch::Tensor &rbot2,
    torch::Tensor &output,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply
    //single_direction=0
);

int corr1d_cuda_backward(
    torch::Tensor &input1,
    torch::Tensor &input2,
    torch::Tensor &rbot1,
    torch::Tensor &rbot2,
    torch::Tensor &gradOutput,
    torch::Tensor &gradInput1,
    torch::Tensor &gradInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply
    // single_direction=0
);
