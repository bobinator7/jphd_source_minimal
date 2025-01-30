// -------------------------------------------------------------------------
// Testbench
// -------------------------------------------------------------------------

`timescale 1ps/1ps

module tb();

    ///// param
    localparam IN_WIDTH = 12;
    localparam MAC_WIDTH = 26;    //IN_WIDTH + COEFF_WIDTH + $clog2(N)
    localparam OUT_WIDTH = 12;
    //localparam COEFF_WIDTH = 12;
    localparam N = 5;
    localparam N_DWT = 4;

    `include "constantsk_ref.svh"
    //localparam N_FC = 11;    

    //localparam CHN_1 = 10;
    //localparam CHN_2 = 13;
    //localparam CHN_3 = 20;
    //localparam CHN_4 = 63;
    //localparam CHN_FC = 4;
    
    // stimulus
    parameter TEST_LEN = 5000;
    
    // tb parameter
    parameter CLK_PRD = 32'd1000000; // 1MHz 
    parameter NB_CYC_RST = 10;
    parameter DATA_IN_PRD = 32'd3333333333; //300Hz=3.33ms    
 
    ///// reg / wire /////
    // system
    reg clk;
    reg rst_n;
    reg in_ready;
    wire out_enable_predict;
    wire [1:0] prediction;

    // stimulus
    reg signed [IN_WIDTH-1:0] x_in_testvec [0:TEST_LEN-1];
    reg [16:0] in_index;
    reg signed [IN_WIDTH-1:0] x_in;

    //
    reg done;
    
    ///// DUT instantiation /////
    cnn_top DUT (
        .clk(clk),
        .rst_n(rst_n),
        .mode_i(1'b0),
        .ecg_ready_i(in_ready),
        .ecg_predict_en_o(out_enable_predict),        
        .ecg_data_i(x_in),
        .ecg_predict_label_o(prediction)
    );
    
    // simulation setup
    `define do_stop 0
	task conditional_stop;
	  if (`do_stop) $stop;
	endtask
      
    ///// generate clock signal /////
    initial begin 
        clk = 1;
        forever begin
            #(CLK_PRD/2) clk = ~clk;
        end 
    end
        
    ///// load stimuli /////
    initial begin 
        $readmemh("../../src/dat_in_ref.txt",x_in_testvec);
    end
    
    ///// debug /////
    initial begin
        rst_n <= 0;
        repeat (NB_CYC_RST) @(negedge clk);
        rst_n <= 1;
    end
    
    initial begin
        done <= 0;
        in_ready <= 0;
        repeat (NB_CYC_RST) @(posedge clk);
        for (in_index = 0; in_index < TEST_LEN; in_index = in_index+1) begin
            in_ready <= 1;
            x_in <= x_in_testvec[in_index];
            repeat (1) @(posedge clk);
            in_ready <= 0;
            repeat (DATA_IN_PRD/CLK_PRD-1) @(posedge clk);
        end
        in_ready <= 1;
        x_in <= 0;
        done <= 1;
        repeat (1) @(posedge clk);
        in_ready <= 0;
		$stop;
    end

//Generating prediction dump
integer pred_BufferFile;

integer pred_nbSample;
initial begin
    pred_BufferFile = $fopen("pred_buffer.txt","w");
end
    
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pred_nbSample <= 0;
    end
    if (out_enable_predict == 1) begin
        pred_nbSample <= pred_nbSample +1;
        if (pred_nbSample > 0)
            $fwrite(pred_BufferFile,"%x\n",prediction);
    end
end

//`ifdef INTERMEDIATE_REF_DUMP
//// count cycles (active compute state, idle state, total)
//integer cyccnt_BufferFile;
//
//integer cyccnt_idle;
//integer cyccnt_active;
//integer cyccnt_nonrst;
//initial begin
//    cyccnt_BufferFile = $fopen("cyccnt_buffer.txt","w");
//end
//    
//always @(posedge clk or negedge rst_n) begin
//    if (!rst_n) begin
//        cyccnt_idle <= 0;
//        cyccnt_active <= 0;
//        cyccnt_nonrst <= 0;
//    end
//    else begin
//        cyccnt_nonrst <= cyccnt_nonrst + 1;
//        if (DUT.main_fsm_state == DUT.MAIN_FSM_STA_IDLE) begin
//            cyccnt_idle <= cyccnt_idle + 1;
//        end
//        if (DUT.main_fsm_state != DUT.MAIN_FSM_STA_IDLE) begin
//            cyccnt_active <= cyccnt_active + 1;
//        end
//        if (done == 1) begin
//            $fwrite(cyccnt_BufferFile,"No. cycles in idle state: %d\n",cyccnt_idle);
//            $fwrite(cyccnt_BufferFile,"No. cycles in active compute state: %d\n",cyccnt_active);
//            $fwrite(cyccnt_BufferFile,"No. cycles in after rst_n not active: %d\n",cyccnt_nonrst);
//        end
//    end
//end
//
////Generating dwt dumps files
//integer data_a1BufferFile; 
//integer data_a2BufferFile;
//integer data_a3BufferFile;
//integer data_a4BufferFile;
//integer data_d4BufferFile;
//
//integer a1_nbSample;
//integer a2_nbSample;
//integer a3_nbSample;
//integer a4_nbSample;
//initial begin
//    data_a1BufferFile = $fopen("data_a1_buffer.txt","w");
//    data_a2BufferFile = $fopen("data_a2_buffer.txt","w");
//    data_a3BufferFile = $fopen("data_a3_buffer.txt","w");
//    data_a4BufferFile = $fopen("data_a4_buffer.txt","w");
//    data_d4BufferFile = $fopen("data_d4_buffer.txt","w");
//end
//
//always @(posedge clk or negedge rst_n) begin
//    if (!rst_n) begin
//        a1_nbSample <= 0;
//        a2_nbSample <= 0;
//        a3_nbSample <= 0;
//        a4_nbSample <= 0;
//    end
//    if (DUT.dwt_nstage_inst.cell_enable[1] == 1) begin
//        a1_nbSample <= a1_nbSample +1;
//        if (a1_nbSample > 0)
//            $fwrite(data_a1BufferFile,"%x\n",DUT.dwt_nstage_inst.cell_out[1]);
//    end
//    if (DUT.dwt_nstage_inst.cell_enable[2] == 1) begin
//        a2_nbSample <= a2_nbSample +1;
//        if (a2_nbSample > 1)
//            $fwrite(data_a2BufferFile,"%x\n",DUT.dwt_nstage_inst.cell_out[2]);
//    end
//    if (DUT.dwt_nstage_inst.cell_enable[3] == 1) begin
//        a3_nbSample <= a3_nbSample +1;
//        if (a3_nbSample > 1)
//            $fwrite(data_a3BufferFile,"%x\n",DUT.dwt_nstage_inst.cell_out[3]);
//    end
//    if (DUT.dwt_nstage_inst.cell_enable[4] == 1) begin
//        a4_nbSample <= a4_nbSample +1;
//        if (a4_nbSample > 1) begin
//            $fwrite(data_a4BufferFile,"%x\n",DUT.dwt_nstage_inst.cell_out[4]);
//            $fwrite(data_d4BufferFile,"%x\n",DUT.dwt_nstage_inst.cell_hp_out);
//        end
//    end
//end
//
//// Generating conv dumps files
//integer conv1_buffer_file;
//integer conv2_buffer_file;
//integer conv3_buffer_file;
//integer conv4_buffer_file;
//
//integer pool1_buffer_file;
//integer pool2_buffer_file;
//integer pool3_buffer_file;
//integer pool4_buffer_file;
//
//integer conv1_nb_samples [0:CHN_1];
//integer conv2_nb_samples [0:CHN_2];
//integer conv3_nb_samples [0:CHN_3];
//integer conv4_nb_samples [0:CHN_4];
//                         
//integer pool1_nb_samples [0:CHN_1];
//integer pool2_nb_samples [0:CHN_2];
//integer pool3_nb_samples [0:CHN_3];
//integer pool4_nb_samples [0:CHN_4];
//
//initial begin
//    conv1_buffer_file = $fopen("conv1_buffer_file.txt","w"); 
//    conv2_buffer_file = $fopen("conv2_buffer_file.txt","w");
//    conv3_buffer_file = $fopen("conv3_buffer_file.txt","w");
//    conv4_buffer_file = $fopen("conv4_buffer_file.txt","w");
//                      
//    pool1_buffer_file = $fopen("pool1_buffer_file.txt","w");
//    pool2_buffer_file = $fopen("pool2_buffer_file.txt","w");
//    pool3_buffer_file = $fopen("pool3_buffer_file.txt","w");
//    pool4_buffer_file = $fopen("pool4_buffer_file.txt","w");
//
//   
//end
//
//integer i;
//always @(posedge clk or negedge rst_n) begin
//    for (i=0; i<CHN_1; i = i+1) begin
//        if (!rst_n) begin
//            conv1_nb_samples[i] <= 0;
//            pool1_nb_samples[i] <= 0;
//        end
//        if (DUT.pool1_en[i] == 1) begin
//            conv1_nb_samples[i] <= conv1_nb_samples[i] +1;
//            if (conv1_nb_samples[i] > 5)
//                $fwrite(conv1_buffer_file,"%x\n", DUT.pe_to_pool1[i]);
//        end
//        if (DUT.next_conv_en[2] == 1) begin
//            pool1_nb_samples[i] <= pool1_nb_samples[i] + 1;
//            if (pool1_nb_samples[i] > 1)
//                $fwrite(pool1_buffer_file,"%x\n",DUT.pool1_to_conv2_fifo[i]);
//        end
//    end
//    for (i=0; i<CHN_2; i = i+1) begin
//        if (!rst_n) begin
//            conv2_nb_samples[i] <= 0;
//            pool2_nb_samples[i] <= 0;
//        end
//        if (DUT.pool2_en[i] == 1) begin
//            conv2_nb_samples[i] <= conv2_nb_samples[i] +1;
//            if (conv2_nb_samples[i] > 5)
//                $fwrite(conv2_buffer_file,"%x\n", DUT.pe_to_pool2[i]);
//        end
//        if (DUT.next_conv_en[3] == 1) begin
//            pool2_nb_samples[i] <= pool2_nb_samples[i] + 1;
//            if (pool2_nb_samples[i] > 1)
//                $fwrite(pool2_buffer_file,"%x\n",DUT.pool2_to_conv3_fifo[i]);
//        end
//    end
//    for (i=0; i<CHN_3; i = i+1) begin
//        if (!rst_n) begin
//            conv3_nb_samples[i] <= 0;
//            pool3_nb_samples[i] <= 0;
//        end
//        if (DUT.pool3_en[i] == 1) begin
//            conv3_nb_samples[i] <= conv3_nb_samples[i] +1;
//            if (conv3_nb_samples[i] > 5)
//                $fwrite(conv3_buffer_file,"%x\n", DUT.pe_to_pool3[i]);
//        end
//        if (DUT.next_conv_en[4] == 1) begin
//            pool3_nb_samples[i] <= pool3_nb_samples[i] + 1;
//            if (pool3_nb_samples[i] > 1)
//                $fwrite(pool3_buffer_file,"%x\n",DUT.pool3_to_conv4_fifo[i]);
//        end
//    end 
//    for (i=0; i<CHN_4; i = i+1) begin
//        if (!rst_n) begin
//            conv4_nb_samples[i] <= 0;
//            pool4_nb_samples[i] <= 0;
//        end
//        if (DUT.pool4_en[i] == 1) begin
//            conv4_nb_samples[i] <= conv4_nb_samples[i] +1;
//            if (conv4_nb_samples[i] > 5)
//                $fwrite(conv4_buffer_file,"%x\n", DUT.pe_to_pool4[i]);
//        end
//        if (DUT.next_fc_en == 1) begin
//            pool4_nb_samples[i] <= pool4_nb_samples[i] + 1;
//            if (pool4_nb_samples[i] > 1)
//                $fwrite(pool4_buffer_file,"%x\n",DUT.pool4_to_fc_fifo[i]);
//        end
//    end
//end
//
//integer fc_buffer_file;
//
//integer fc_nb_samples [0:CHN_FC];
//initial begin
//    fc_buffer_file = $fopen("fc_buffer_file.txt","w");
//end
//
//always @(posedge clk or negedge rst_n) begin
//    for (i = 0; i < CHN_FC; i = i+1) begin
//        if (!rst_n)
//            fc_nb_samples[i] <= 0;
//        if (DUT.pred_en_reg[i]) begin
//            fc_nb_samples[i] = fc_nb_samples[i] + 1;
//            if (fc_nb_samples[i] > 10)
//                $fwrite(fc_buffer_file, "%x\n",DUT.pred[i]);
//        end
//    end
//end
//`endif
endmodule

