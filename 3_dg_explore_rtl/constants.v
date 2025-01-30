// Main FSM
localparam MAIN_FSM_STA_IDLE = 3'b000;
localparam MAIN_FSM_STA_CONV1 = 3'b001;
localparam MAIN_FSM_STA_CONV2 = 3'b010;
localparam MAIN_FSM_STA_CONV3 = 3'b011;
localparam MAIN_FSM_STA_CONV4 = 3'b100;
localparam MAIN_FSM_STA_FC = 3'b101;
localparam MAIN_FSM_STA_MEMWR = 3'b110;

localparam MAIN_FSM_STA_CL = 3'b111;

//// NN fixed rom param
//localparam CHN_DWT = 12'd2;
//localparam CHN_1 = 12'd10;
//localparam CHN_2 = 12'd13;
//localparam CHN_3 = 12'd20;
//localparam CHN_4 = 12'd63;
//localparam CHN_FC = 12'd4;
//
//localparam N_FC = 11;

