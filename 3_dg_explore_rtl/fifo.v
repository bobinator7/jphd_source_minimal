// ------------------------------------------------------------------------------------
// Title            : Fifo
// Project          : FWVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : fifo.v
// Author           : Louis Cherel
// Company          : IDS RWTH Aachen
// Created          : 2020/10/23
// ------------------------------------------------------------------------------------
// Description      : This module implements a fifo buffer with parallel access to every 
//                    element based on shift registers.
// ------------------------------------------------------------------------------------
// Copyright by IDS 2020
// ------------------------------------------------------------------------------------

// TODO
// - check if replacable by ip blocks

module cnn_fifo #( 
    parameter IN_WIDTH = 12,
    parameter N = 5 // length of the kernel
)(
    input clk,
    input rst_n,

    input [IN_WIDTH-1:0] x_in,
    input  in_enable,

    output [IN_WIDTH*N-1:0] x_out
);

reg [IN_WIDTH-1:0] fifo_cell [0:N-1];

genvar i;
generate 
    for (i = 1; i < N; i = i + 1) begin : fifo
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                fifo_cell[i] <= 'b0;
            end
            else if (in_enable == 1) begin
                fifo_cell[i] <= fifo_cell[i-1];
            end
        end
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fifo_cell[0] <= 'b0;
    end
    else if (in_enable == 1) begin
        fifo_cell[0] <= x_in;
    end
end

genvar j;
generate 
    for (j=0; j<N; j=j+1) begin : output_assign
        assign x_out[j*IN_WIDTH +: IN_WIDTH] = fifo_cell[j];
    end
endgenerate

endmodule











