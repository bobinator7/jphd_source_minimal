// ------------------------------------------------------------------------------------
// Title            : SRAM Memory for WVCNN parameters
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : nn_param_ram.v
// Author           : Johnson Loh
// Company          : IDS RWTH Aachen
// Created          : 2021/01/29
// ------------------------------------------------------------------------------------
// Description      : Wrapper for SRAM macro returning weight kernels and biases based
//                    on no. input/output channel and in case of FC layers no. of
//                    sample within time sequence
// ------------------------------------------------------------------------------------
// Copyright by IDS 2021
// ------------------------------------------------------------------------------------

module nn_param_ram #(
    parameter WIDTH = 12,
    parameter MEM_ADR_MAX_WIDTH = 13,
    parameter N = 5 // length of the kernel
)(
    // system
    input clk,
    input rst_n,

    // io control
    input [2:0] main_fsm_state,

    // mem config
    input mem_ram_dest_i,
    input [MEM_ADR_MAX_WIDTH-1:0] mem_ram_adr_i,
    input [5-1:0] mem_ram_adr_offset_i,
    input [WIDTH-1:0] mem_ram_data_i,

    // io
    input [9:0] ch_in_i,
    input [9:0] ch_out_i,
    output [N*WIDTH-1:0] weight_o,
    output [WIDTH-1:0] bias_o
);

/// param ///
`include "constants.v"
`include "constantsk_ref.svh"

//// mem derived param
//localparam OFFSET_WEIGHT_CHN_1 = 12'd0;
//localparam OFFSET_WEIGHT_CHN_2 = CHN_1*CHN_DWT+OFFSET_WEIGHT_CHN_1;
//localparam OFFSET_WEIGHT_CHN_3 = CHN_2*CHN_1+OFFSET_WEIGHT_CHN_2;
//localparam OFFSET_WEIGHT_CHN_4 = CHN_3*CHN_2+OFFSET_WEIGHT_CHN_3;
//localparam OFFSET_WEIGHT_CHN_FC = CHN_4*CHN_3+OFFSET_WEIGHT_CHN_4;
//
//localparam OFFSET_BIAS_CHN_1 = 7'd0;
//localparam OFFSET_BIAS_CHN_2 = CHN_1+OFFSET_BIAS_CHN_1;
//localparam OFFSET_BIAS_CHN_3 = CHN_2+OFFSET_BIAS_CHN_2;
//localparam OFFSET_BIAS_CHN_4 = CHN_3+OFFSET_BIAS_CHN_3;
//localparam OFFSET_BIAS_CHN_FC = CHN_4+OFFSET_BIAS_CHN_4;

/// reg / wire ///
wire mem_cen_b;
wire [13-1:0] weight_ram_inst_a;
wire [60-1:0] weight_ram_inst_q;
wire [8-1:0] bias_ram_inst_a;
wire [16-1:0] bias_ram_inst_q;

reg [13-1:0] weight_a_offset;
reg [13-1:0] weight_a_chout_mul;
reg [8-1:0] bias_a_offset;

wire weight_ram_inst_wren;
wire bias_ram_inst_wren;

//
wire [60-1:0] ram_inst_bw;
wire [60-1:0] ram_inst_d;
wire ram_inst_deepsleep;
wire ram_inst_powergate;

/// modules ///
// weight RAM
MBH_ZSNL_IN22FDX_S1C_NFRG_W06144B060M16C128 weight_ram_inst (
    .clk(clk),
    .cen(mem_cen_b),
    .rdwen(weight_ram_inst_wren),
    .deepsleep(ram_inst_deepsleep),
    .powergate(ram_inst_powergate),
    .a(weight_ram_inst_a), //[2+7+4-1:0]
    .d(ram_inst_d),
    .bw(ram_inst_bw),
    .q(weight_ram_inst_q) //[60-1:0]
);

// bias RAM
MBH_ZSNL_IN22FDX_S1C_NFRG_W00256B016M08C128 bias_ram_inst (
    .clk(clk),
    .cen(mem_cen_b),
    .rdwen(bias_ram_inst_wren),
    .deepsleep(ram_inst_deepsleep),
    .powergate(ram_inst_powergate),
    .a(bias_ram_inst_a), //[1+4+3-1:0]
    .d(ram_inst_d[16-1:0]),
    .bw(ram_inst_bw[16-1:0]),
    .q(bias_ram_inst_q) //[16-1:0]
);

/// comb ///
always @(*) begin
    case(main_fsm_state)
        MAIN_FSM_STA_CONV1 : begin
            weight_a_offset <= WEIGHT_OFFSET[0];
            bias_a_offset <= BIAS_OFFSET[0];

            weight_a_chout_mul <= BLOCK_CHN[0];
        end

        MAIN_FSM_STA_CONV2 : begin
            weight_a_offset <= WEIGHT_OFFSET[1];
            bias_a_offset <= BIAS_OFFSET[1];

            weight_a_chout_mul <= BLOCK_CHN[1];
        end

        MAIN_FSM_STA_CONV3 : begin
            weight_a_offset <= WEIGHT_OFFSET[2];
            bias_a_offset <= BIAS_OFFSET[2];

            weight_a_chout_mul <= BLOCK_CHN[2];
        end

        MAIN_FSM_STA_CONV4 : begin
            weight_a_offset <= WEIGHT_OFFSET[3];
            bias_a_offset <= BIAS_OFFSET[3];

            weight_a_chout_mul <= BLOCK_CHN[3];
        end

        MAIN_FSM_STA_FC : begin
            weight_a_offset <= WEIGHT_OFFSET[4];
            bias_a_offset <= BIAS_OFFSET[4];

            weight_a_chout_mul <= FC_K*BLOCK_CHN[4]/N;
        end

        default begin
            weight_a_offset <= 12'd0;
            bias_a_offset <= 7'd0;

            weight_a_chout_mul <= 12'd0;
        end
    endcase
end

/// assigns ///
assign ram_inst_bw = {{(60-WIDTH){1'b0}},{(WIDTH){1'b1}}} << WIDTH*mem_ram_adr_offset_i;
assign ram_inst_d = {{(60-WIDTH){1'b0}},{mem_ram_data_i}} << WIDTH*mem_ram_adr_offset_i;
assign ram_inst_deepsleep = 1'b0;
assign ram_inst_powergate = 1'b0;

assign mem_wr = (main_fsm_state != MAIN_FSM_STA_MEMWR); //active low
assign mem_cen_b = (main_fsm_state == MAIN_FSM_STA_IDLE) & mem_wr;

assign weight_ram_inst_wren = mem_ram_dest_i ? mem_wr : 1'b1;
assign bias_ram_inst_wren = (~mem_ram_dest_i) ? mem_wr : 1'b1;

assign weight_ram_inst_a = (mem_wr == 0) ? mem_ram_adr_i : (ch_out_i*weight_a_chout_mul + ch_in_i + weight_a_offset);
assign bias_ram_inst_a = (mem_wr == 0) ? mem_ram_adr_i : (ch_out_i + bias_a_offset);

assign weight_o = (!mem_cen_b) ? weight_ram_inst_q : {(N*WIDTH){1'b0}};
assign bias_o = (!mem_cen_b) ? bias_ram_inst_q[12-1:0] : {(WIDTH){1'b0}};

endmodule
