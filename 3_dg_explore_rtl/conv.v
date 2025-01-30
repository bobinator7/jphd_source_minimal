// ------------------------------------------------------------------------------------
// Title            : 1D-convolution module
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : conv.v
// Author           : Johnson Loh
// Company          : IDS RWTH Aachen
// Created          : 2021/01/14
// ------------------------------------------------------------------------------------
// Description      : Convolution (Comb-only)
// ------------------------------------------------------------------------------------
// Copyright by IDS 2021
// ------------------------------------------------------------------------------------
module conv #( 
    parameter IN_WIDTH = 12,
    parameter WEIGHT_WIDTH = 12,
    parameter MAC_WIDTH = 26, // > IN_WIDTH + COEFF_WIDTH + $clog2(N)
    parameter OUT_WIDTH = 12,
    parameter FRA_WIDTH = 8, // fractional length of the quantization scheme
    parameter N = 5 // length of the kernel
)(
    // system
    input clk,
    input rst_n,

    // io control
    //input in_enable,

    // io
    input [N*IN_WIDTH-1:0] x_in,
    input [N*WEIGHT_WIDTH-1:0] h_in,
    output signed [OUT_WIDTH-1:0] y_out
);

/// reg / wire ///
// io
wire signed [IN_WIDTH-1:0] x_w [0:N-1];
wire signed [WEIGHT_WIDTH-1:0] h_w [0:N-1];

// MAC
wire signed [(IN_WIDTH + WEIGHT_WIDTH)-1:0] mul_line [0:N-1];
wire signed [MAC_WIDTH-1:0] sum_line [0:N-1];


/// assigns ///
// mul_line
genvar l;
generate
    for (l = 0; l < N; l = l + 1) begin : mul
        assign x_w[l] = x_in[l*IN_WIDTH+:IN_WIDTH];
        assign h_w[l] = h_in[(N-l-1)*WEIGHT_WIDTH+:WEIGHT_WIDTH];
        assign mul_line[l] = x_w[l] * h_w[l];
    end
endgenerate

// sum_line
assign sum_line[0] = mul_line[0];
genvar k;
generate
    for (k = 1; k < N; k = k + 1) begin : sum
        assign sum_line[k] = sum_line[k-1] + mul_line[k];
    end
endgenerate

// output
assign y_out = sum_line[N-1][FRA_WIDTH+:OUT_WIDTH] + sum_line[N-1][FRA_WIDTH-1]; //y_out_reg;

endmodule
