// ------------------------------------------------------------------------------------
// Title            : CNN top level design
// Project          : WVCNN accelerator (Gen. 2)
// ------------------------------------------------------------------------------------
// File             : top.v
// Author           : Johnson Loh
// Company          : IDS RWTH Aachen
// Created          : 2020/12/18
// ------------------------------------------------------------------------------------
// Description      : This module implements the DWT and convolutional neural network
//                    used for the classification of ECG signals. It builds on the full
//                    flat design from doi:10.1109/ASAP49362.2020.00042, but seperating
//                    the computation units from the pipelined memory.
//                    Forked from the ECG FPGA Prototype
// ------------------------------------------------------------------------------------
// Copyright by IDS 2021
// ------------------------------------------------------------------------------------

module cnn_top #(
    parameter IN_WIDTH = 12,
    parameter OUT_WIDTH = 12,
    parameter COEFF_WIDTH = 12,
    parameter FRA_WIDTH = 8, // fractional length of the quantization scheme

    parameter N = 5, // length of the convolution kernel
    parameter MAC_WIDTH = 26, // > IN_WIDTH + COEFF_WIDTH + $clog2(N)
    parameter MAC_WIDTH_FC = 23,
    parameter ACC_WIDTH = 20, // IN_WIDTH + log2(MAX_CHN_COUNT)

    parameter MEM_ADR_MAX_WIDTH = 13,

    parameter Nb_MAC = 1 //number of MAC unit
)(
    // system
    input clk,
    input rst_n,

    // mode control
    input mode_i, //0: compute, 1: init

    // NN memory
    input mem_ram_dest_i, //0: bias, 1: weights
    input [MEM_ADR_MAX_WIDTH-1:0] mem_ram_adr_i,
    input [5-1:0] mem_ram_adr_offset_i,
    input mem_ram_data_en_i,
    input [IN_WIDTH-1:0] mem_ram_data_i,

    //// Debug pins
    //output [2:0] debug_state_o,
    //output debug_dwt_insampl_o,
    //output debug_dwt_outsampl_o,
    //output debug_cnn_pred_en_o,
    //output debug_cnn_predval_valid_o,

    //// Debug rm
    //output [4:0] debug_conv_en,
    //output [10-1:0] debug_pool1_en,
    //output [13-1:0] debug_pool2_en,
    //output [20-1:0] debug_pool3_en,
    //output [63-1:0] debug_pool4_en,
    //output debug_fc_en,
    //output [4-1:0] debug_pred_en,
    //output [2*IN_WIDTH-1:0] debug_dwt_data,
    //output [10*IN_WIDTH-1:0] debug_conv1_data,
    //output [13*IN_WIDTH-1:0] debug_conv2_data,
    //output [20*IN_WIDTH-1:0] debug_conv3_data,
    //output [63*IN_WIDTH-1:0] debug_conv4_data,
    //output [4*IN_WIDTH-1:0] debug_fc_data,
    //output [10*OUT_WIDTH-1:0] debug_pool1_data,
    //output [13*OUT_WIDTH-1:0] debug_pool2_data,
    //output [20*OUT_WIDTH-1:0] debug_pool3_data,
    //output [63*OUT_WIDTH-1:0] debug_pool4_data,

    // ECG
    input ecg_ready_i,
    output ecg_predict_en_o,
    input signed [IN_WIDTH-1:0] ecg_data_i,
    output [1:0] ecg_predict_label_o
);

// ---------- global params ----------
`include "constants.v"
`include "constantsk_cl.svh"

localparam CYC_FC = 14;//$ceil(FC_K*BLOCK_CHN[5]/N);
localparam CYC_CL = 5;//$ceil(BLOCK_CHN[2]/N);

// ---------- reg & wires ----------
//// Main FSM
reg [2:0] main_fsm_state;
reg [2:0] main_fsm_next_state;

wire conv_state;
wire fc_state;
wire active_state;

wire conv_next_state;
wire fc_next_state;
wire active_next_state;

//// CONV control
// conv_ch_in counter control
wire conv_ch_in_cnt_en;
reg [9:0] conv_ch_in_cnt;
reg [9:0] conv_ch_in_cnt_dly1;
reg [10:0] conv_ch_in_cnt_max;

// conv_ch_out counter control
wire conv_ch_out_cnt_en;
reg [9:0] conv_ch_out_cnt;
reg [10:0] conv_ch_out_cnt_max;

// convolution subroutine control
wire conv_start;
wire conv_ch_in_first_cyc;
wire conv_ch_in_last_cyc;
reg conv_ch_in_last_cyc_dly1, conv_ch_in_last_cyc_dly2;
wire conv_ch_out_last_cyc;
wire conv_cnt_done;

// fc subroutine control
wire conv_fc_done;

// accumulator control
wire add_en_w;
reg add_en;

wire add_new_bias;

// pooling control
reg [9:0] pool_in_cnt;
reg pool_en;
wire pool_done;
reg pool_done_reg;

wire pool_last_cyc;
wire pool_cnt_en;
reg pool_cnt_valid;
//wire pool_start;

// enable next layer computation
wire [5:1] next_conv_en;
wire next_fc_en;

//// DWT mem
wire dwt_nstage_inst_in_enable;
wire signed [OUT_WIDTH-1:0] dwt_nstage_inst_a_out;
wire signed [OUT_WIDTH-1:0] dwt_nstage_inst_d_out;

//// FIFO mem
wire [N*IN_WIDTH-1:0] conv1_in [0:BLOCK_CHN[0]-1];
wire [N*IN_WIDTH-1:0] conv2_in [0:BLOCK_CHN[1]-1];
wire [N*IN_WIDTH-1:0] conv3_in [0:BLOCK_CHN[3]-1];
wire [N*IN_WIDTH-1:0] conv4_in [0:BLOCK_CHN[4]-1];
wire [FC_K*IN_WIDTH-1:0] fc_in [0:BLOCK_CHN[5]-1];
wire [IN_WIDTH-1:0] fc_in_vec [0:FC_K*BLOCK_CHN[5]-1];
wire [N*IN_WIDTH-1:0] fc_data_sel_in [0:CYC_FC-1];

// predict register
//TODO
wire pred_en [0:BLOCK_CHN[6]-1];
//reg [BLOCK_CHN[6]-1:0] pred_en_reg;
reg [OUT_WIDTH-1:0] pred [0:BLOCK_CHN[6]-1];

reg [0:BLOCK_CHN[6]-1] out_en_reg;

//// WEIGHT/BIAS mem
wire [N*IN_WIDTH-1:0] nn_param_mem_inst_weight_data;
wire [IN_WIDTH-1:0] nn_param_mem_inst_bias_data;

//// PE array
wire [OUT_WIDTH-1:0] conv_inst_result_out [0:Nb_MAC-1];

//// mem to pe interconnect
// lvl pe
reg [N*IN_WIDTH-1:0] pe_array_conv_data_in [0:Nb_MAC-1];
wire [N*COEFF_WIDTH-1:0] pe_array_conv_weight_in [0:Nb_MAC-1];
wire [COEFF_WIDTH-1:0] pe_array_conv_bias_in [0:Nb_MAC-1];

wire [OUT_WIDTH-1:0] pe_array_accum_out [0:Nb_MAC-1];

// lvl mem
reg [N*IN_WIDTH-1:0] conv1_to_pe [0:Nb_MAC-1];
reg [N*IN_WIDTH-1:0] conv2_to_pe [0:Nb_MAC-1];
reg [N*IN_WIDTH-1:0] conv3_to_pe [0:Nb_MAC-1];
reg [N*IN_WIDTH-1:0] conv4_to_pe [0:Nb_MAC-1];
reg [N*IN_WIDTH-1:0] fc_to_pe [0:Nb_MAC-1];

wire [BLOCK_CHN[1]-1:0] pool1_en;
wire [BLOCK_CHN[2]-1:0] pool2_en;
wire [BLOCK_CHN[4]-1:0] pool3_en;
wire [BLOCK_CHN[5]-1:0] pool4_en;
wire [BLOCK_CHN[1]-1:0] pool1_out_en;
wire [BLOCK_CHN[2]-1:0] pool2_out_en;
wire [BLOCK_CHN[4]-1:0] pool3_out_en;
wire [BLOCK_CHN[5]-1:0] pool4_out_en;
reg [OUT_WIDTH-1:0] pe_to_pool1 [0:BLOCK_CHN[1]-1];
reg [OUT_WIDTH-1:0] pe_to_pool2 [0:BLOCK_CHN[2]-1];
reg [OUT_WIDTH-1:0] pe_to_pool3 [0:BLOCK_CHN[4]-1];
reg [OUT_WIDTH-1:0] pe_to_pool4 [0:BLOCK_CHN[5]-1];
reg [OUT_WIDTH-1:0] pe_to_pred [0:BLOCK_CHN[6]-1];

//// mem to mem interconnect
wire [OUT_WIDTH-1:0] pool1_to_conv2_fifo [0:BLOCK_CHN[1]-1];
wire [OUT_WIDTH-1:0] pool2_to_cl_fifo [0:BLOCK_CHN[2]-1];
wire [OUT_WIDTH-1:0] pool3_to_conv4_fifo [0:BLOCK_CHN[4]-1];
wire [OUT_WIDTH-1:0] pool4_to_fc_fifo [0:BLOCK_CHN[5]-1];

// cl additions
//TODO: cl_en next_conv_en
wire [IN_WIDTH-1:0] cl_in [0:BLOCK_CHN[2]-1];
wire [BLOCK_CHN[3]-1:0] cl_en;
reg cl_en_reg;
reg [N*IN_WIDTH-1:0] cl_to_pe [0:Nb_MAC-1];
reg [OUT_WIDTH-1:0] pe_to_conv3 [0:BLOCK_CHN[3]-1];
wire [N*IN_WIDTH-1:0] cl_data_sel_in [0:CYC_CL-1];
//wire [IN_WIDTH-1:0] cl_in_vec [0:BLOCK_CHN[3]-1];


// ---------- modules ----------
//// DWT modules
dwt_nstage #(
    .IN_WIDTH(IN_WIDTH),
    .COEFF_WIDTH(COEFF_WIDTH),
    .OUT_WIDTH(OUT_WIDTH),
    .FRA_WIDTH(FRA_WIDTH),
    .MAC_WIDTH(MAC_WIDTH),
    .DEPTH(4),
    .DWT_INIT({1'b0,1'b0,1'b1,1'b0})
) dwt_nstage_inst (
    .clk(clk),
    .rst_n(rst_n),
    .in_enable(dwt_nstage_inst_in_enable),
    .out_enable(next_conv_en[1]),

    .x_in(ecg_data_i),
    .dwt_a_out(dwt_nstage_inst_a_out),
    .dwt_d_out(dwt_nstage_inst_d_out)
);

//// CONV1 input buffer
// opt. TODO: vectorize
cnn_fifo #(
    .IN_WIDTH(IN_WIDTH),
    .N(N)
) CONV1_fifo_A (
    .clk(clk),
    .rst_n(rst_n),
    .x_in(dwt_nstage_inst_a_out),
    .in_enable(next_conv_en[1]),
    .x_out(conv1_in[0])
);

cnn_fifo #(
    .IN_WIDTH(IN_WIDTH),
    .N(N)
) CONV1_fifo_D (
    .clk(clk),
    .rst_n(rst_n),
    .x_in(dwt_nstage_inst_d_out),
    .in_enable(next_conv_en[1]),
    .x_out(conv1_in[1])
);

//// genvars
genvar i,j;

//// POOL1
generate
    for (i=0; i<BLOCK_CHN[1]; i = i+1) begin : P1_FIFO
        maxpool #(
            .BIT_WIDTH(IN_WIDTH),
            .N(3),
            .POOL_INIT(2)
        ) pool_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_enable(pool1_en[i]),
            .out_enable(pool1_out_en[i]),
            .data_in(pe_to_pool1[i]),
            .data_out(pool1_to_conv2_fifo[i])
        );
    end
endgenerate

//// CONV2 input buffer
generate
    for (i=0; i<BLOCK_CHN[1]; i = i+1) begin : CONV2_input_fifo
        cnn_fifo #(
            .IN_WIDTH(IN_WIDTH),
            .N(N)
        ) fifo_inst (
            .clk(clk),
            .rst_n(rst_n),
            .x_in(pool1_to_conv2_fifo[i]),
            .in_enable(next_conv_en[2]),
            .x_out(conv2_in[i])
        );
    end
endgenerate

//// POOL2
generate
    for (i=0; i<BLOCK_CHN[2]; i = i+1) begin : P2_FIFO
        maxpool #(
            .BIT_WIDTH(IN_WIDTH),
            .N(3),
            .POOL_INIT(2)
        ) pool_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_enable(pool2_en[i]),
            .out_enable(pool2_out_en[i]),
            .data_in(pe_to_pool2[i]),
            .data_out(pool2_to_cl_fifo[i])
        );
    end
endgenerate

//CL Buffer 
generate
    for (i=0; i<BLOCK_CHN[2]; i = i+1) begin : CL_input_fifo
        cnn_fifo #(
            .IN_WIDTH(IN_WIDTH),
            .N(1) 
        ) fifo_inst (
            .clk(clk),
            .rst_n(rst_n),
            .x_in(pool2_to_cl_fifo[i]),
            .in_enable(next_conv_en[3]),
            .x_out(cl_in[i])
        );
    end
endgenerate

//// CONV3 input buffer
generate
    for (i=0; i<BLOCK_CHN[3]; i = i+1) begin : CONV3_input_fifo
        cnn_fifo #(
            .IN_WIDTH(IN_WIDTH),
            .N(N)
        ) fifo_inst (
            .clk(clk),
            .rst_n(rst_n),
            .x_in(pe_to_conv3[i]),
            //.in_enable(next_conv_en[4]),
            .in_enable(cl_en[i]),
            .x_out(conv3_in[i])
        );
    end
endgenerate

//// POOL3
generate
    for (i=0; i<BLOCK_CHN[4]; i = i+1) begin : P3_FIFO
        maxpool #(
            .BIT_WIDTH(IN_WIDTH),
            .N(3),
            .POOL_INIT(2)
        ) pool_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_enable(pool3_en[i]),
            .out_enable(pool3_out_en[i]),
            .data_in(pe_to_pool3[i]),
            .data_out(pool3_to_conv4_fifo[i])
        );
    end
endgenerate

//// CONV4 input buffer
generate
    for (i=0; i<BLOCK_CHN[4]; i = i+1) begin : CONV4_input_fifo
        cnn_fifo #(
            .IN_WIDTH(IN_WIDTH),
            .N(N)
        ) fifo_inst (
            .clk(clk),
            .rst_n(rst_n),
            .x_in(pool3_to_conv4_fifo[i]),
            .in_enable(next_conv_en[5]),
            .x_out(conv4_in[i])
        );
    end
endgenerate

//// POOL4
generate
    for (i=0; i<BLOCK_CHN[5]; i = i+1) begin : P4_FIFO
        maxpool #(
            .BIT_WIDTH(IN_WIDTH),
            .N(3),
            .POOL_INIT(2)
        ) pool_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_enable(pool4_en[i]),
            .out_enable(pool4_out_en[i]),
            .data_in(pe_to_pool4[i]),
            .data_out(pool4_to_fc_fifo[i])
        );
    end
endgenerate

//// FC input buffer
generate
    for (i=0; i<BLOCK_CHN[5]; i = i+1) begin : FC_input_fifo
        cnn_fifo #(
            .IN_WIDTH(IN_WIDTH),
            .N(FC_K)
        ) fc_fifo (
            .clk(clk),
            .rst_n(rst_n),
            .x_in(pool4_to_fc_fifo[i]),
            .in_enable(next_fc_en),
            .x_out(fc_in[i])
        );
    end
endgenerate

//// WEIGHT/BIAS RAM
// size for weight_o and bias_o supported for Nb_MAC = 1 only
// replace ROM with width WIDTH*N*Nb_MAC for other configurations
// SRAM data port width <256 -> max. N=4 possible
nn_param_ram #(
    .WIDTH(12),
    .MEM_ADR_MAX_WIDTH(13),
    .N(5) // length of the kernel
) nn_param_ram_inst (
    .clk(clk),
    .rst_n(rst_n),
    .main_fsm_state(main_fsm_state),
    .mem_ram_dest_i(mem_ram_dest_i),
    .mem_ram_adr_i(mem_ram_adr_i),
    .mem_ram_adr_offset_i(mem_ram_adr_offset_i),
    .mem_ram_data_i(mem_ram_data_i),
    .ch_in_i(conv_ch_in_cnt),
    .ch_out_i(conv_ch_out_cnt),
    .weight_o(nn_param_mem_inst_weight_data),
    .bias_o(nn_param_mem_inst_bias_data)
);

//// CONV PE array
generate
    for (i=0; i<Nb_MAC; i=i+1) begin : PE_ARRAY
        conv #(
            .IN_WIDTH(IN_WIDTH),
            .WEIGHT_WIDTH(COEFF_WIDTH),

            .MAC_WIDTH(MAC_WIDTH),
            .OUT_WIDTH(OUT_WIDTH),
            .FRA_WIDTH(FRA_WIDTH),
            .N(N)
        ) conv_inst (

            .clk(clk),
            .rst_n(rst_n),
            .x_in(pe_array_conv_data_in[i]),
            .h_in(pe_array_conv_weight_in[i]),
            .y_out(conv_inst_result_out[i])
        );

        add #(
            .BIT_WIDTH(IN_WIDTH),
            .BIAS_WIDTH(IN_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) add_inst (
            .clk(clk),
            .rst_n(rst_n),
            .in_enable(add_en),
            .new_bias(add_new_bias),
            .x_in(conv_inst_result_out[i]),
            .bias_in(pe_array_conv_bias_in[i]),
            .y_out_relu(pe_array_accum_out[i])
        );
    end
endgenerate

// ---------- FSM -----------
//// Main FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        main_fsm_state <= MAIN_FSM_STA_IDLE;
    else
        main_fsm_state <= main_fsm_next_state;
end

always @(*) begin
    case(main_fsm_state)
        MAIN_FSM_STA_IDLE : begin
            if ((mem_ram_data_en_i & mode_i) == 1)
                main_fsm_next_state = MAIN_FSM_STA_MEMWR;
            else if (next_conv_en[1] == 1)
                main_fsm_next_state = MAIN_FSM_STA_CONV1;
            else
                main_fsm_next_state = main_fsm_state;
        end

        MAIN_FSM_STA_CONV1 : begin
            if (pool_done_reg == 1)
                if (next_conv_en[2] == 1)
                    main_fsm_next_state = MAIN_FSM_STA_CONV2;
                else
                    main_fsm_next_state = MAIN_FSM_STA_IDLE;
            else
                main_fsm_next_state = main_fsm_state;
        end

        MAIN_FSM_STA_CONV2 : begin
            if (pool_done_reg == 1)
                if (next_conv_en[3] == 1)
                    main_fsm_next_state = MAIN_FSM_STA_CL;
                else
                    main_fsm_next_state = MAIN_FSM_STA_IDLE;
            else
                main_fsm_next_state = main_fsm_state;
        end

        MAIN_FSM_STA_CL : begin
            if (next_conv_en[4] == 1)
                main_fsm_next_state = MAIN_FSM_STA_CONV3;
            else
                main_fsm_next_state = main_fsm_state;
        end

        MAIN_FSM_STA_CONV3 : begin
            if (pool_done_reg == 1)
                if (next_conv_en[5] == 1)
                    main_fsm_next_state = MAIN_FSM_STA_CONV4;
                else
                    main_fsm_next_state = MAIN_FSM_STA_IDLE;
            else
                main_fsm_next_state = main_fsm_state;
        end

        MAIN_FSM_STA_CONV4 : begin
            if (pool_done_reg == 1)
                if (next_fc_en == 1)
                    main_fsm_next_state = MAIN_FSM_STA_FC;
                else
                    main_fsm_next_state = MAIN_FSM_STA_IDLE;
            else
                main_fsm_next_state = main_fsm_state;
        end


        MAIN_FSM_STA_FC : begin
            if (conv_fc_done == 1)
                main_fsm_next_state = MAIN_FSM_STA_IDLE;
            else
                main_fsm_next_state = main_fsm_state;
        end

        MAIN_FSM_STA_MEMWR : begin
            main_fsm_next_state = MAIN_FSM_STA_IDLE;
        end


        default begin
            main_fsm_next_state = MAIN_FSM_STA_IDLE;
        end
    endcase
end

//// Seq Logic
/// TODO: move cyclic counter logic to seperate module
// counter to adress the input channel for convolution
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        conv_ch_in_cnt <= 10'd0;
    end
    else if (conv_ch_in_cnt_en == 1) begin
        if (conv_ch_in_last_cyc) begin
            conv_ch_in_cnt <= 10'd0;
        end
        else begin
            conv_ch_in_cnt <= conv_ch_in_cnt + 10'd1;
        end
    end
end

// counter to adress the first output channel of the PE batch for convolution
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        conv_ch_out_cnt <= 10'd0;
    end
    else if (conv_ch_out_cnt_en == 1) begin
        if (conv_ch_out_last_cyc) begin
            conv_ch_out_cnt <= 10'd0;
        end
        else begin
            conv_ch_out_cnt <= conv_ch_out_cnt + Nb_MAC;
        end
    end
end

//
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        conv_ch_in_cnt_dly1 <= 10'd0;
    end
    else begin
        if (conv_start & conv_cnt_done) begin
            conv_ch_in_cnt_dly1 <= 10'd0;
        end
        else begin
            conv_ch_in_cnt_dly1 <= conv_ch_in_cnt;
        end
    end
end

// registers to indicate cycle after computation of a batch of output channels has finished for pooling computation (1 cycle ROM delay, 1 cycle output register delay)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        conv_ch_in_last_cyc_dly1 <= 1'b0;
    end
    else begin
        conv_ch_in_last_cyc_dly1 <= conv_ch_in_last_cyc;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        conv_ch_in_last_cyc_dly2 <= 1'b0;
    end
    else begin
        conv_ch_in_last_cyc_dly2 <= conv_ch_in_last_cyc_dly1;
    end
end

//
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pool_cnt_valid <= 1'b0;
    end
    else begin
        if ((conv_ch_in_last_cyc_dly1 == 0) & (conv_ch_in_last_cyc_dly2 == 1)) begin
            pool_cnt_valid <= 1'b1;
        end
        else if (pool_last_cyc) begin
            pool_cnt_valid <= 1'b0;
        end
        else begin
            pool_cnt_valid <= pool_cnt_valid;
        end
    end
end

// pool enable triggered by posedge of conv_ch_in_last_cyc
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pool_en <= 1'b0;
    end
    else begin
        pool_en <= (conv_ch_in_last_cyc_dly1 == 1) & (conv_ch_in_last_cyc_dly2 == 0);
    end
end

// delay reg for cl layer
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cl_en_reg <= 1'b0;
    end
    else begin
        cl_en_reg <= cl_en[BLOCK_CHN[3]-1];
    end
end

// max counter logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        conv_ch_in_cnt_max <= 11'd0;
        conv_ch_out_cnt_max <= 11'd0;
    end
    else begin
        case(main_fsm_next_state)
            MAIN_FSM_STA_CONV1 : begin
                conv_ch_in_cnt_max <= BLOCK_CHN[0]-1;
                conv_ch_out_cnt_max <= BLOCK_CHN[1]-1;
            end

            MAIN_FSM_STA_CONV2 : begin
                conv_ch_in_cnt_max <= BLOCK_CHN[1]-1;
                conv_ch_out_cnt_max <= BLOCK_CHN[2]-1;
            end

            MAIN_FSM_STA_CL : begin
                conv_ch_in_cnt_max <= 5-1;//$ceil(BLOCK_CHN[2]/N)-1;
                conv_ch_out_cnt_max <= BLOCK_CHN[3]-1;
            end

            MAIN_FSM_STA_CONV3 : begin
                conv_ch_in_cnt_max <= BLOCK_CHN[3]-1;
                conv_ch_out_cnt_max <= BLOCK_CHN[4]-1;
            end

            MAIN_FSM_STA_CONV4 : begin
                conv_ch_in_cnt_max <= BLOCK_CHN[4]-1;
                conv_ch_out_cnt_max <= BLOCK_CHN[5]-1;
            end

            MAIN_FSM_STA_FC : begin
                conv_ch_in_cnt_max <= 14-1;//$ceil(BLOCK_CHN[5]*FC_K/N)-1;
                conv_ch_out_cnt_max <= BLOCK_CHN[6]-1;
            end

            default begin
                conv_ch_in_cnt_max <= 11'd0;
                conv_ch_out_cnt_max <= 11'd0;
            end
        endcase
    end
end

// enable accumulator
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        add_en <= 1'b0;
    end
    else begin
        add_en <= add_en_w;
    end
end

/// pooling
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pool_in_cnt <= 10'd0;
    end
    else if (pool_cnt_en == 1) begin
        if (pool_last_cyc | conv_start) begin
            pool_in_cnt <= 10'd0;
        end
        else begin
            pool_in_cnt <= pool_in_cnt + Nb_MAC;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pool_done_reg <= 1'b0;
    end
    else begin
        pool_done_reg <= pool_done;
    end
end

/// Predict buffer
// predict register (0: normal, 1: af, 2: others, 3: noise)
generate
    for (i=0; i<BLOCK_CHN[6]; i = i+1) begin : pred_data_reg
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                pred[i] <= {(OUT_WIDTH){1'b0}};
            end
            //else if (pred_en_reg[i]) begin
            else if (pred_en[i]) begin
                pred[i] <= pe_to_pred[i];
            end
        end
    end
endgenerate

//// predict enable
//generate
//    for (i=0; i<BLOCK_CHN[6]; i = i+1) begin : pred_enable_reg
//        always @(posedge clk or negedge rst_n) begin
//            if (!rst_n) begin
//                pred_en_reg[i] <= 1'b0;
//            end
//            else begin
//                pred_en_reg[i] <= pred_en[i];
//            end
//        end
//    end
//endgenerate

/// Out enable registers
generate
    for (i=0; i<BLOCK_CHN[6]; i = i+1) begin : out_enable_reg
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                out_en_reg[i] <= 1'b0;
            end
            else begin
                if (active_state) begin
                    //if (pred_en_reg[i]) begin
                    if (pred_en[i]) begin
                        out_en_reg[i] <= 1'b1;
                    end
                    else begin
                        out_en_reg[i] <= out_en_reg[i];
                    end
                end
                else begin
                    out_en_reg[i] <= 1'b0;
                end
            end
        end
    end
endgenerate

//// Comb Logic
/// mem to pe interconnect (lvl pe)
generate
    for (i=0; i<Nb_MAC; i = i+1) begin : conv_data_in_mux
        always @(*) begin
            case(main_fsm_state)
                MAIN_FSM_STA_CONV1 : begin
                    if (i >= BLOCK_CHN[1]) begin
                        pe_array_conv_data_in[i] = {(N*IN_WIDTH){1'b0}};
                    end
                    else begin
                        pe_array_conv_data_in[i] = conv1_to_pe[i];
                    end
                end


                MAIN_FSM_STA_CONV2 : begin
                    if (i >= BLOCK_CHN[2]) begin
                        pe_array_conv_data_in[i] = {(N*IN_WIDTH){1'b0}};
                    end
                    else begin
                        pe_array_conv_data_in[i] = conv2_to_pe[i];
                    end
                end

                MAIN_FSM_STA_CL : begin
                    if (i >= BLOCK_CHN[3]) begin
                        pe_array_conv_data_in[i] = {(N*IN_WIDTH){1'b0}};
                    end
                    else begin
                        pe_array_conv_data_in[i] = cl_to_pe[i];
                    end
                end

                MAIN_FSM_STA_CONV3 : begin
                    if (i >= BLOCK_CHN[4]) begin
                        pe_array_conv_data_in[i] = {(N*IN_WIDTH){1'b0}};
                    end
                    else begin
                        pe_array_conv_data_in[i] = conv3_to_pe[i];
                    end
                end

                MAIN_FSM_STA_CONV4 : begin
                    if (i >= BLOCK_CHN[5]) begin
                        pe_array_conv_data_in[i] = {(N*IN_WIDTH){1'b0}};
                    end
                    else begin
                        pe_array_conv_data_in[i] = conv4_to_pe[i];
                    end
                end

                MAIN_FSM_STA_FC : begin
                    pe_array_conv_data_in[i] = fc_to_pe[i];
                end

                default begin
                    pe_array_conv_data_in[i] = {(N*IN_WIDTH){1'b0}};
                end
            endcase
        end
    end
endgenerate


/// mem to pe interconnect (lvl mem)
// to pe mux
generate
    for (i=0; i<Nb_MAC; i = i+1) begin : fifo_data_out_mux
        always @(*) begin
            conv1_to_pe[i] = conv1_in[conv_ch_in_cnt_dly1];

            conv2_to_pe[i] = conv2_in[conv_ch_in_cnt_dly1];

            conv3_to_pe[i] = conv3_in[conv_ch_in_cnt_dly1];

            conv4_to_pe[i] = conv4_in[conv_ch_in_cnt_dly1];

            //cl_to_pe[i] = cl_in[conv_ch_in_cnt_dly1];
            //cl_to_pe[i] = {cl_data_sel_in[0],cl_data_sel_in[1],cl_data_sel_in[2],cl_data_sel_in[3],cl_data_sel_in[4]}; //TODO
            //cl_to_pe[i] = {cl_data_sel_in[4],cl_data_sel_in[3],cl_data_sel_in[2],cl_data_sel_in[1],cl_data_sel_in[0]}; //TODO
            cl_to_pe[i] = cl_data_sel_in[conv_ch_in_cnt_dly1];

            //fc_to_pe[i] = {fc_data_sel_in[0],fc_data_sel_in[1],fc_data_sel_in[2],fc_data_sel_in[3],fc_data_sel_in[4]}; //TODO
            fc_to_pe[i] = {fc_data_sel_in[4],fc_data_sel_in[3],fc_data_sel_in[2],fc_data_sel_in[1],fc_data_sel_in[0]}; //TODO
            //fc_to_pe[i] = fc_data_sel_in[conv_ch_in_cnt_dly1];

        end
    end
endgenerate

// to mem demux
generate
    for (i=0; i<BLOCK_CHN[1]; i = i+1) begin : pool1_data_in_mux
        always @(*) begin
            if (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) begin
                pe_to_pool1[i] = pe_array_accum_out[i-pool_in_cnt];
            end
            else begin
                pe_to_pool1[i] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

generate
    for (i=0; i<BLOCK_CHN[2]; i = i+1) begin : pool2_data_in_mux
        always @(*) begin
            if (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) begin
                pe_to_pool2[i] = pe_array_accum_out[i-pool_in_cnt];
            end
            else begin
                pe_to_pool2[i] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

generate
    for (i=0; i<BLOCK_CHN[3]; i = i+1) begin : cl_data_in_mux
        always @(*) begin
            if (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) begin
                pe_to_conv3[i] = pe_array_accum_out[i-pool_in_cnt];
            end
            else begin
                pe_to_conv3[i] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

generate
    for (i=0; i<BLOCK_CHN[4]; i = i+1) begin : pool3_data_in_mux
        always @(*) begin
            if (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) begin
                pe_to_pool3[i] = pe_array_accum_out[i-pool_in_cnt];
            end
            else begin
                pe_to_pool3[i] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

generate
    for (i=0; i<BLOCK_CHN[5]; i = i+1) begin : pool4_data_in_mux
        always @(*) begin
            if (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) begin
                pe_to_pool4[i] = pe_array_accum_out[i-pool_in_cnt];
            end
            else begin
                pe_to_pool4[i] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

generate
    for (i=0; i<BLOCK_CHN[6]; i = i+1) begin : pred_data_in_mux
        always @(*) begin
            if (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) begin
                pe_to_pred[i] = pe_array_accum_out[i-pool_in_cnt];
            end
            else begin
                pe_to_pred[i] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

//// Assigns
// dwt control
assign dwt_nstage_inst_in_enable = ecg_ready_i & ~mode_i;

//
assign pe_array_conv_weight_in[0] = nn_param_mem_inst_weight_data;
assign pe_array_conv_bias_in[0] = nn_param_mem_inst_bias_data;

// layer-specific pool enable
generate
    for (i=0; i<BLOCK_CHN[1]; i = i+1) begin : pool1_en_mux
        assign pool1_en[i] = (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) ? pool_en & (main_fsm_state == MAIN_FSM_STA_CONV1) : 1'b0;
    end
endgenerate
generate
    for (i=0; i<BLOCK_CHN[2]; i = i+1) begin : pool2_en_mux
        assign pool2_en[i] = (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) ? pool_en & (main_fsm_state == MAIN_FSM_STA_CONV2) : 1'b0;
    end
endgenerate
generate
    for (i=0; i<BLOCK_CHN[4]; i = i+1) begin : pool3_en_mux
        assign pool3_en[i] = (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) ? pool_en & (main_fsm_state == MAIN_FSM_STA_CONV3) : 1'b0;
    end
endgenerate
generate
    for (i=0; i<BLOCK_CHN[5]; i = i+1) begin : pool4_en_mux
        assign pool4_en[i] = (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) ? pool_en & (main_fsm_state == MAIN_FSM_STA_CONV4) : 1'b0;
    end
endgenerate
generate
    for (i=0; i<BLOCK_CHN[3]; i = i+1) begin : cl_en_mux
        assign cl_en[i] = (i >= pool_in_cnt && i < pool_in_cnt+Nb_MAC) ? pool_en & (main_fsm_state == MAIN_FSM_STA_CL) : 1'b0;
    end
endgenerate

// vectorize fc input
generate
    for (i=0; i<BLOCK_CHN[5]; i = i+1) begin : fc_in_vectorized
        for (j=0; j<FC_K; j = j+1) begin : fc_in_vectorized
            assign fc_in_vec[i*FC_K+(FC_K-1-j)] = fc_in[i][j*(IN_WIDTH)+:IN_WIDTH];
        end
    end
endgenerate

// fc pre-select
generate
    for (i=0; i<N; i = i+1) begin : fc_data_vec_sel_in
        assign fc_data_sel_in[i] = (conv_ch_in_cnt_dly1*N+i > conv_ch_in_cnt_max) ? {(IN_WIDTH){1'b0}} : fc_in_vec[conv_ch_in_cnt_dly1*N+i];
    end
endgenerate
//generate
//    for (i=0; i<CYC_FC; i = i+1) begin : fc_data_vec_sel_in
//        for (j=0; j<N; j = j+1) begin : fc_data_vec_sel_in2
//            if ((N*i+j) < (BLOCK_CHN[4]*FC_K)) begin
//                assign fc_data_sel_in[i][j*IN_WIDTH+:IN_WIDTH] = fc_in_vec[N*i+j];
//            end else begin
//                assign fc_data_sel_in[i][j*IN_WIDTH+:IN_WIDTH] = {(IN_WIDTH){1'b0}};
//            end
//        end
//    end
//endgenerate

//// cl pre-select
//generate
//    for (i=0; i<N; i = i+1) begin : cl_data_vec_sel_in
//        assign cl_data_sel_in[i] = (conv_ch_in_cnt_dly1*N+i > conv_ch_in_cnt_max) ? {(IN_WIDTH){1'b0}} : cl_in[conv_ch_in_cnt_dly1*N+i];
//    end
//endgenerate
generate
    for (i=0; i<CYC_CL; i = i+1) begin : cl_data_vec_sel_in
        for (j=0; j<N; j = j+1) begin : cl_data_vec_sel_in2
            if ((N*i+j) < BLOCK_CHN[2]) begin
                assign cl_data_sel_in[i][j*IN_WIDTH+:IN_WIDTH] = cl_in[N*i+j];
            end else begin
                assign cl_data_sel_in[i][j*IN_WIDTH+:IN_WIDTH] = {(IN_WIDTH){1'b0}};
            end
        end
    end
endgenerate

// signals indicating current type of layer
assign conv_state = (main_fsm_state == MAIN_FSM_STA_CONV1) | (main_fsm_state == MAIN_FSM_STA_CONV2) | (main_fsm_state == MAIN_FSM_STA_CONV3) | (main_fsm_state == MAIN_FSM_STA_CONV4) | (main_fsm_state == MAIN_FSM_STA_CL);
assign fc_state = (main_fsm_state == MAIN_FSM_STA_FC);
assign active_state = conv_state | fc_state;

assign conv_next_state = (main_fsm_next_state == MAIN_FSM_STA_CONV1) | (main_fsm_next_state == MAIN_FSM_STA_CONV2) | (main_fsm_next_state == MAIN_FSM_STA_CONV3) | (main_fsm_next_state == MAIN_FSM_STA_CONV4) | (main_fsm_next_state == MAIN_FSM_STA_CL);
assign fc_next_state = (main_fsm_next_state == MAIN_FSM_STA_FC);
assign active_next_state = conv_next_state | fc_next_state;

// trigger signals to start of convolution subroutines
assign conv_start = (main_fsm_state != main_fsm_next_state) & active_next_state;
assign conv_ch_in_cnt_en = conv_start | (active_state & ~conv_cnt_done); //TAG
assign conv_ch_out_cnt_en = conv_start | (active_state & conv_ch_in_last_cyc & ~conv_ch_out_last_cyc); //TAG

// trigger signals indicating end of convolution subroutines
assign conv_ch_in_first_cyc = (conv_ch_in_cnt == 0);
assign conv_ch_in_last_cyc = (conv_state) ? (conv_ch_in_cnt >= conv_ch_in_cnt_max) : ((conv_ch_in_cnt * N) > conv_ch_in_cnt_max); // multiplication cnt*N should be synthesized as constant propagation (shift + add)
assign conv_ch_out_last_cyc = ((conv_ch_out_cnt + Nb_MAC) > conv_ch_out_cnt_max);
assign conv_cnt_done = conv_ch_in_last_cyc & conv_ch_out_last_cyc;

// trigger signals indicating start/end of pooling subroutines
//assign pool_start = (conv_ch_in_last_cyc_dly1 == 1) & (conv_ch_in_last_cyc_dly2 == 0) & ~pool_cnt_valid;
assign pool_last_cyc = ((pool_in_cnt + Nb_MAC) > conv_ch_out_cnt_max);
assign pool_cnt_en = (pool_en & ~pool_last_cyc) | conv_start;
assign pool_done = conv_ch_in_last_cyc & pool_last_cyc & pool_en;

// accumulator control
assign add_en_w = (active_state & ((~(conv_ch_in_last_cyc_dly1 & conv_ch_out_last_cyc)) | pool_cnt_valid )); //TAG
assign add_new_bias = ((conv_ch_in_cnt_dly1 == 0) & active_state); //TAG

// predict register control
generate
    for (i=0; i<BLOCK_CHN[6]; i = i+1) begin : pred_en_ctrl
        assign pred_en[i] = (fc_state & conv_ch_in_last_cyc & (conv_ch_out_cnt == i)); //TAG
    end
endgenerate

// trigger signals to change states in fsm
assign next_conv_en[2] = &pool1_out_en & pool_done_reg & (main_fsm_state == MAIN_FSM_STA_CONV1);
assign next_conv_en[3] = &pool2_out_en & pool_done_reg & (main_fsm_state == MAIN_FSM_STA_CONV2);
assign next_conv_en[4] = cl_en_reg & (main_fsm_state == MAIN_FSM_STA_CL);
assign next_conv_en[5] = &pool3_out_en & pool_done_reg & (main_fsm_state == MAIN_FSM_STA_CONV3);
assign next_fc_en = &pool4_out_en & pool_done_reg & (main_fsm_state == MAIN_FSM_STA_CONV4);

assign conv_fc_done = &out_en_reg & (main_fsm_state == MAIN_FSM_STA_FC);

// output
assign ecg_predict_en_o = conv_fc_done;
assign ecg_predict_label_o[0] = (pred[0] > pred[1]) ;//& (pred[0] > pred[2]) & (pred[0] > pred[3]);
assign ecg_predict_label_o[1] = (pred[1] > pred[0]) ;//& (pred[1] > pred[2]) & (pred[1] > pred[3]);
//assign ecg_predict_label_o[2] = (pred[2] > pred[1]) & (pred[2] > pred[0]) & (pred[2] > pred[3]);
//assign ecg_predict_label_o[3] = (pred[3] > pred[1]) & (pred[3] > pred[2]) & (pred[3] > pred[0]);

//// debug
//assign debug_state_o = main_fsm_state;
//assign debug_dwt_insampl_o = dwt_nstage_inst_in_enable;
//assign debug_dwt_outsampl_o = next_conv_en[1];
//assign debug_cnn_pred_en_o = conv_fc_done;
//assign debug_cnn_predval_valid_o = ((ecg_predict_label_o[0] ^ ecg_predict_label_o[1]) ^ (ecg_predict_label_o[2] ^ ecg_predict_label_o[3]));
//
//assign debug_conv_en[4:1] = next_conv_en;
//assign debug_conv_en[0] = dwt_nstage_inst_in_enable;
//assign debug_pool1_en = pool1_en;
//assign debug_pool2_en = pool2_en;
//assign debug_pool3_en = pool3_en;
//assign debug_pool4_en = pool4_en;
//assign debug_fc_en = next_fc_en;
//assign debug_pred_en = pred_en_reg;
//assign debug_dwt_data[0*IN_WIDTH+:IN_WIDTH] = dwt_nstage_inst_a_out;
//assign debug_dwt_data[1*IN_WIDTH+:IN_WIDTH] = dwt_nstage_inst_d_out;
////assign debug_conv1_data = pe_to_pool1;
////assign debug_conv2_data = pe_to_pool2;
////assign debug_conv3_data = pe_to_pool3;
////assign debug_conv4_data = pe_to_pool4;
////assign debug_fc_data = pe_to_pred;
////assign debug_pool1_data = pool1_to_conv2_fifo;
////assign debug_pool2_data = pool2_to_cl_fifo;
////assign debug_pool3_data = pool3_to_conv4_fifo;
////assign debug_pool4_data = pool4_to_fc_fifo;
//
//generate
//    for (i=0; i<10; i = i+1) begin : debug_conv1_data_gen
//        assign debug_conv1_data[i*IN_WIDTH+:IN_WIDTH] = pe_to_pool1[i];
//    end
//endgenerate
//generate
//    for (i=0; i<13; i = i+1) begin : debug_conv2_data_gen
//        assign debug_conv2_data[i*IN_WIDTH+:IN_WIDTH] = pe_to_pool2[i];
//    end
//endgenerate
//generate
//    for (i=0; i<20; i = i+1) begin : debug_conv3_data_gen
//        assign debug_conv3_data[i*IN_WIDTH+:IN_WIDTH] = pe_to_pool3[i];
//    end
//endgenerate
//generate
//    for (i=0; i<63; i = i+1) begin : debug_conv4_data_gen
//        assign debug_conv4_data[i*IN_WIDTH+:IN_WIDTH] = pe_to_pool4[i];
//    end
//endgenerate
//generate
//    for (i=0; i<4; i = i+1) begin : debug_fc_data_gen
//        assign debug_fc_data[i*IN_WIDTH+:IN_WIDTH] = pe_to_pred[i];
//    end
//endgenerate
//
//generate
//    for (i=0; i<10; i = i+1) begin : debug_pool1_data_gen
//        assign debug_pool1_data[i*IN_WIDTH+:IN_WIDTH] = pool1_to_conv2_fifo[i];
//    end
//endgenerate
//generate
//    for (i=0; i<13; i = i+1) begin : debug_pool2_data_gen
//        assign debug_pool2_data[i*IN_WIDTH+:IN_WIDTH] = pool2_to_cl_fifo[i];
//    end
//endgenerate
//generate
//    for (i=0; i<20; i = i+1) begin : debug_pool3_data_gen
//        assign debug_pool3_data[i*IN_WIDTH+:IN_WIDTH] = pool3_to_conv4_fifo[i];
//    end
//endgenerate
//generate
//    for (i=0; i<63; i = i+1) begin : debug_pool4_data_gen
//        assign debug_pool4_data[i*IN_WIDTH+:IN_WIDTH] = pool4_to_fc_fifo[i];
//    end
//endgenerate

endmodule
