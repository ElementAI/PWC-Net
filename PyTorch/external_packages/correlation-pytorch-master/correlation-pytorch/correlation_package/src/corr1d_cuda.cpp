#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "corr1d_cuda_kernel.h"

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// == Forward
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
)
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(rbot1);
    CHECK_INPUT(rbot2);
    CHECK_INPUT(output);
    // TODO: Shapechecks

    int batchSize = input1.size(0);

    long nInputPlane = input1.size(1);
    long nInputRows = input1.size(2);
    long nInputCols = input1.size(3);
    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    int x_shift = -neighborhood_grid_radius_;

    // Number of output channels amounts to displacement combinations in X direction only!!
    int nOutputPlane = neighborhood_grid_width_;//Same, because 1D X-correlation 

    // Inputs
    float * input1_data = input1.data<float>();
    float * input2_data = input2.data<float>();

    // Outputs
    output.resize_({batchSize, nOutputPlane, nOutputRows, nOutputCols});
    output.zero_(); // added by Jinwei
    float * output_data = output.data<float>();

    rbot1.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});
    rbot2.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});

    rbot1.zero_(); // added by Jinwei
    rbot2.zero_(); // added by Jinwei

    float * rbot1_data = rbot1.data<float>();
    float * rbot2_data = rbot2.data<float>();

    auto stream  = at::cuda::getCurrentCUDAStream().stream();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    blob_rearrange_ongpu_1d(input1_data,rbot1_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    blob_rearrange_ongpu_1d(input2_data,rbot2_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    CorrelateData_ongpu_1d(rbot1_data,rbot2_data,output_data,batchSize,nOutputCols,nOutputRows,nOutputPlane,max_displacement,x_shift,neighborhood_grid_width_,kernel_radius_,kernel_size,stride1,stride2,paddedbottomwidth,paddedbottomheight,nInputPlane,corr_type_multiply,stream);

//    THCudaTensor_free(state, input1);
//    THCudaTensor_free(state, input2);
//    THCudaTensor_free(state, rbot1);
//    THCudaTensor_free(state, rbot2);

    return 1;
}

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
)
{
    float * input1_data = input1.data<float>();
    float * input2_data = input2.data<float>();

    long nInputCols = input1.size(3);
    long nInputRows = input1.size(2);
    long nInputPlane = input1.size(1);
    long batchSize = input1.size(0);

 //   THCudaTensor_resizeAs(state, gradInput1, input1);
 //   THCudaTensor_resizeAs(state, gradInput2, input2);
    float * gradOutput_data = gradOutput.data<float>();
    float * gradInput1_data = gradInput1.data<float>();
    float * gradInput2_data = gradInput2.data<float>();

    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    int x_shift = -neighborhood_grid_radius_;

    // Number of output channels amounts to displacement combinations in X direction only!!
    int nOutputPlane = neighborhood_grid_width_; // Same, because 1D X-correlation

    rbot1.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});
    rbot2.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});

    rbot1.zero_(); // added by Jinwei
    rbot2.zero_(); // added by Jinwei

    float * rbot1_data = rbot1.data<float>();
    float * rbot2_data = rbot2.data<float>();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    auto stream  = at::cuda::getCurrentCUDAStream().stream();

    blob_rearrange_ongpu_1d(input1_data,rbot1_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    blob_rearrange_ongpu_1d(input2_data,rbot2_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight,stream);

    // CorrelationLayerBackward

    CorrelateDataBackward_ongpu_1d(rbot1_data,rbot2_data,gradOutput_data,gradInput1_data,gradInput2_data,batchSize,nOutputCols,nOutputRows,nOutputPlane,max_displacement,x_shift,neighborhood_grid_width_,kernel_radius_,stride1,stride2,nInputCols,nInputRows,paddedbottomwidth,paddedbottomheight,nInputPlane,pad_size,corr_type_multiply,stream);

  //  THCudaTensor_free(state, input1);
  //  THCudaTensor_free(state, input2);
  //  THCudaTensor_free(state, rbot1);
  //  THCudaTensor_free(state, rbot2);

    return 1;

}