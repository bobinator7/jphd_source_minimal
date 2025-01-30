// ------------------------------------------------------------------------------------
// Title            : Accumulator with bias reset
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : add.v
// Author           : Louis Cherel
// Company          : IDS RWTH Aachen
// Created          : 2020/12/08
// ------------------------------------------------------------------------------------
// Description      : Accumulates input over each cycle and resettable with constant
// ------------------------------------------------------------------------------------
// Copyright by IDS 2020
// ------------------------------------------------------------------------------------
module add #( 
    parameter BIT_WIDTH = 12,
    parameter BIAS_WIDTH = 12,
    parameter ACC_WIDTH = 17 //BIAS_WIDTH + $clog2(CHN_MAX)
)(
    // system
    input clk,
    input rst_n,

    // io control
    input in_enable,
    input new_bias,

    //data
    input signed [BIT_WIDTH-1:0] x_in,
    input signed [BIAS_WIDTH-1:0] bias_in,
    output signed [BIT_WIDTH-1:0] y_out_relu
);

// ---------- reg & wires ----------
reg signed [ACC_WIDTH-1:0] acc;
wire signed [BIT_WIDTH-1:0] y_out;

// ---------- Sequential logic ----------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc <= 'b0;
    end
    else if (in_enable == 1'b1) begin
        if (new_bias == 1) begin
            acc <= x_in + bias_in;
        end
        else begin
            acc <= x_in + acc;
        end
    end
end

//Positive Saturation
assign y_out = (|acc[ACC_WIDTH-1:BIT_WIDTH-1] == 1) ? 12'h7FF : acc[BIT_WIDTH-1:0];
assign y_out_relu = (acc[ACC_WIDTH-1]==0)? y_out : 0 ; //relu

endmodule
