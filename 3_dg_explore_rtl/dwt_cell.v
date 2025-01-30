// ------------------------------------------------------------------------------------
// Title            : DWT_cell
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : dwt_cell.v
// Author           : Johnson Loh (Jianan Wen)
// Company          : IDS RWTH Aachen
// Created          : 2020/04/01
// ------------------------------------------------------------------------------------
// Description      : DWT one stage - local cnt for subsampling
// ------------------------------------------------------------------------------------
// Copyright by IDS 2021
// ------------------------------------------------------------------------------------
module dwt_cell #( 
    parameter IN_WIDTH = 12,
    parameter COEFF_WIDTH = 12,
    parameter MAC_WIDTH = 26,    // > IN_WIDTH + COEFF_WIDTH + $clog2(N)
    parameter OUT_WIDTH = 12,
    parameter FRA_WIDTH = 8,  // fractional length of the quantization scheme
    parameter N = 4,  // length of the wavelet function
    parameter [COEFF_WIDTH*N -1 :0] H_IN = {12'h7C,12'hD6,12'h39,12'hFDF},
    parameter DWT_INIT = 1'b0  // initialize initial cnt value to match subsampling pattern
)(
    // system
    input clk,
    input rst_n,

    // io control
    input in_enable,
    output out_enable,

    // io
    input signed [IN_WIDTH-1:0] x_in,
    output signed [OUT_WIDTH-1:0] y_out
);

/// reg / wire ///
// control
reg dwt_cyc_cnt;
reg dwt_cyc_cnt_dly1;
wire out_enable_w;

// io
wire signed [COEFF_WIDTH-1:0] h_wire [0:N-1];
reg signed [OUT_WIDTH-1:0] y_out_reg;

// MAC
reg  signed [IN_WIDTH-1:0] tap_line [0:N-1];
wire signed [(IN_WIDTH + COEFF_WIDTH)-1:0] mul_line [0:N-1];
wire signed [MAC_WIDTH-1:0] sum_line [0:N-1];

/// seq logic ///
// dwt_cyc_cnt logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dwt_cyc_cnt <= DWT_INIT;
    end
    else if(in_enable == 1) begin
        dwt_cyc_cnt <= dwt_cyc_cnt + 1'b1;
    end
end

// dwt_cyc_cnt_dly1 logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        dwt_cyc_cnt_dly1 <= 1'b0;
    end
    else begin
        dwt_cyc_cnt_dly1 <= dwt_cyc_cnt;
    end
end

// tap_line logic
always @(*) begin
    tap_line[0] <= x_in;
end

genvar j;
generate
    for (j = 1; j < N; j = j + 1) begin : tap
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                tap_line[j] <= 'b0;
            end
            else if (in_enable == 1) begin
                tap_line[j] <= tap_line[j-1];
            end
        end
    end
endgenerate

// y_out_reg logic (round-to-nearest)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        y_out_reg <= 'b0;
    end
    else if (dwt_cyc_cnt == 1 && in_enable == 1) begin
        y_out_reg <= sum_line[N-1][FRA_WIDTH+:OUT_WIDTH] + sum_line[N-1][FRA_WIDTH -1];
    end
end

/// assigns ///
// TODO: store in local registers
genvar i;
generate
    for (i = 0; i < N; i = i + 1) begin : assign_h
        assign h_wire[i] = H_IN[i*COEFF_WIDTH +: COEFF_WIDTH];
    end
endgenerate

// mul_line
genvar l;
generate
    for (l = 0; l < N; l = l + 1) begin : mul
        assign mul_line[l] = tap_line[l] * h_wire[N-l-1];
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

// out_enable_w
assign out_enable_w = (dwt_cyc_cnt_dly1 == 1 & dwt_cyc_cnt == 0) ? 1'b1 : 1'b0;

// out_enable
assign out_enable = out_enable_w;

// y_out
assign y_out = y_out_reg;

endmodule
