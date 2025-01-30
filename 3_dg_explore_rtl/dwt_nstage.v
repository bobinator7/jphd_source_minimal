// ------------------------------------------------------------------------------------
// Title            : N-stage DWT
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : dwt_nstage.v
// Author           : Johnson Loh
// Company          : IDS RWTH Aachen
// Created          : 2020/01/14
// ------------------------------------------------------------------------------------
// Description      : 
// ------------------------------------------------------------------------------------
// Copyright by IDS 2021
// ------------------------------------------------------------------------------------

module dwt_nstage #( 
    parameter IN_WIDTH = 12,
    parameter OUT_WIDTH = 12,
    parameter COEFF_WIDTH = 12,
    parameter FRA_WIDTH = 8, // fractional length of the quantization scheme 
    parameter MAC_WIDTH = 26, // > IN_WIDTH + COEFF_WIDTH + $clog2(N)    

    parameter DEPTH = 4,
    parameter [DEPTH-1:0] DWT_INIT = {1'b0,1'b1,1'b0,1'b0}
)(
    // system
    input clk,
    input rst_n,
    
    // io control
    input in_enable,
    output out_enable,

    // io
    input signed [IN_WIDTH-1:0] x_in,
    output [OUT_WIDTH-1:0] dwt_a_out,
    output [OUT_WIDTH-1:0] dwt_d_out
);

/// params ///
localparam N_DWT = 4;
`include "param_DWT.v"

/// reg / wire ///
wire cell_enable [0:DEPTH];
wire signed [OUT_WIDTH-1:0] cell_out [0:DEPTH];
wire signed [OUT_WIDTH-1:0] cell_hp_out; 

/// modules ///
genvar i;
generate
    for (i=0;i<DEPTH;i=i+1) begin : LP_cells
        dwt_cell #(
            .IN_WIDTH(IN_WIDTH), 
            .COEFF_WIDTH(COEFF_WIDTH), 
            .MAC_WIDTH(MAC_WIDTH), 
            .OUT_WIDTH(OUT_WIDTH), 
            .FRA_WIDTH(FRA_WIDTH), 
            .H_IN(H_IN_LP),
            .N(N_DWT),
            .DWT_INIT(DWT_INIT[i])
        ) dwt_cell_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_enable(cell_enable[i]),
            .out_enable(cell_enable[i+1]),
            .x_in(cell_out[i]), 
            .y_out(cell_out[i+1])
        ); 
    end
endgenerate

dwt_cell #(
    .IN_WIDTH(IN_WIDTH), 
    .COEFF_WIDTH(COEFF_WIDTH), 
    .MAC_WIDTH(MAC_WIDTH), 
    .OUT_WIDTH(OUT_WIDTH), 
    .FRA_WIDTH(FRA_WIDTH), 
    .H_IN(H_IN_HP),
    .N(N_DWT),
    .DWT_INIT(DWT_INIT[DEPTH-1])
) dwt_cell_hp_inst (
    .clk(clk), 
    .rst_n(rst_n),
    .in_enable(cell_enable[DEPTH-1]),
    .out_enable(),
    .x_in(cell_out[DEPTH-1]),
    .y_out(cell_hp_out)
); 

/// assigns ///
assign cell_enable[0] = in_enable;
assign cell_out[0] = x_in;

assign out_enable = cell_enable[DEPTH];
assign dwt_a_out = cell_out[DEPTH];
assign dwt_d_out = cell_hp_out;

endmodule
