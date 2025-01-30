// ------------------------------------------------------------------------------------
// Title            : Sequential maxpooling
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : maxpool.v
// Author           : Johnson Loh
// Company          : IDS RWTH Aachen
// Created          : 2021/01/15
// ------------------------------------------------------------------------------------
// Description      : Sequential maxpooling operation
// ------------------------------------------------------------------------------------
// Copyright by IDS 2021
// ------------------------------------------------------------------------------------
module maxpool #(     
    parameter BIT_WIDTH = 12,
    parameter N = 3, // kernel size of maxpooling
    parameter POOL_INIT = 0 // initialize initial cnt value to match subsampling pattern
)(
    // system
    input clk,
    input rst_n,

    // io control
    input in_enable,
    output out_enable,

    //data
    input [BIT_WIDTH-1:0] data_in,
    output [BIT_WIDTH-1:0] data_out
);

/// params ///
localparam RESET_VAL = {(BIT_WIDTH){1'b0}};

/// reg / wire ///
// control
reg [1:0] pool_cyc_cnt;
wire pool_last_cyc;

// internal var
wire [BIT_WIDTH-1:0] current_max;
reg [BIT_WIDTH-1:0] current_max_reg;
wire [BIT_WIDTH-1:0] next_max;
reg in_enable_reg;

/// seq logic ///
// TODO: move cyclic counter logic to seperate module
// pool_cyc_cnt logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pool_cyc_cnt <= POOL_INIT;
    end
    else if(in_enable == 1) begin
        if (pool_last_cyc) begin
            pool_cyc_cnt <= 'b0;
        end
        else begin
            pool_cyc_cnt <= pool_cyc_cnt + 1'b1;
        end
    end
end

// current_max_reg logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_max_reg <= 'b0;
    end
    else if(in_enable == 1) begin
        current_max_reg <= next_max;
    end
end

// out_enable logic for pulse generation
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_enable_reg <= 1'b0;
    end
    else begin
        if(in_enable == 1) begin
            in_enable_reg <= 1'b1;
        end
        else begin
            in_enable_reg <= 1'b0;
        end
    end
end

/// assigns ///
assign current_max = pool_last_cyc ? RESET_VAL : current_max_reg ;
assign next_max = (data_in > current_max) ? data_in : current_max;
assign pool_last_cyc = ((pool_cyc_cnt + 1'b1) >= N);

assign out_enable = pool_last_cyc; //& in_enable_reg;
assign data_out = current_max_reg;

endmodule
