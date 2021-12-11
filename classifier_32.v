`timescale 1ns / 1ps

module classifier(
    // AXI stream clock and reset signals;
    input wire  M_AXIS_ACLK,
    input wire  M_AXIS_ARESETN,
    input wire  S_AXIS_ACLK,
	input wire  S_AXIS_ARESETN,
    // Master channel signals, write data to DRAM;
    output wire  M_AXIS_TVALID,
    output wire [31:0] M_AXIS_TDATA, 
    output wire  M_AXIS_TLAST, 
    input wire  M_AXIS_TREADY,
    // Slave channel signals, read data from DRAM;
    output wire  S_AXIS_TREADY,
    input wire [31:0] S_AXIS_TDATA, 
    input wire  S_AXIS_TLAST,
    input wire  S_AXIS_TVALID
);

    // Classifier configurations;
    parameter num_bits = 32, q_bits = 8; 
    parameter num_frame = 16, fft_point = 128, fft_point_full = 256, hidden_feature = 32, input_feature = 16, output_feature = 3;
        
    integer k;
    // FSM state register
    reg [3:0] state;
    parameter ldw1 = 4'b0000; // 0000: initial state, wait for w1;
    parameter ldb1 = 4'b0001; // 0001: wait for b1;
    parameter ldw2 = 4'b0010; // 0010: wait for w2
    parameter ldb2 = 4'b0011; // 0011: wait for b2
    parameter ldin = 4'b0100; // 0100: idle state, reset counters when no data is read, wait and process the streaming data;
    parameter fft = 4'b0101; // 0101: fft process;
    parameter maxpool = 4'b0110; // 0110: max pooling;
    parameter mlp = 4'b0111; // 0111: MLP processing;
    parameter datawb = 4'b1000; // 1000: read data from pl;
    parameter rd_maxpool = 4'b1001; // 1001: read data from input_x1;
    parameter rd_x2 = 4'b1010; // 1010: read data from input_x2;


    // Reserved signals for fft module;
    // Control signals;
    reg [31:0] cnt_fft_frame; // Counter to record the output frame;
    reg [31:0] cnt_fft_point; // Counter to record the output point of current frame;

    // Inputs to fft module;
    reg in_valid_fft; // Input valid signal for one sample;
    wire out_valid_fft_flag; // Flag that indicates the output is valid, seems only high for 1 clk for each frame;
    reg out_valid_fft;
    reg [num_bits - 1:0] input_fft;

    // Outputs from fft module;
    wire [41:0] output_fft; // Output containing stft results;
    reg valid_fft; // valid signal of whole fft process, stft data is ready;

    // Reserved signals for max pooling module;
    // Control signals;
    reg [2:0] s_maxpool; // State register in mlp processing;
    parameter maxpool_lnp = 3'b000; // Load a frame from stft data and process;
    parameter maxpool_p = 3'b001; // Pure processing, also write results to mlp inputs;

    // Control registers for max pooling processing;
    reg [31:0] cnt_comp_result; // Counter of comparison array results;


    // Inputs to max pooling;
    reg in_valid_maxpool; //Input valid signal;
    reg out_valid_maxpool; 
    reg [num_bits - 1:0] output_maxpool; // Output of the comparison array;


    reg [num_bits - 1:0] comp0 [0:127];
    reg [num_bits - 1:0] comp1 [0:63];
    reg [num_bits - 1:0] comp2 [0:31];
    reg [num_bits - 1:0] comp3 [0:15];
    reg [num_bits - 1:0] comp4 [0:7];
    reg [num_bits - 1:0] comp5 [0:3];
    reg [num_bits - 1:0] comp6 [0:1];
    reg [5:0] sr_maxpool; // Comparison array has a delay of 8 clks;

    // Binary comparison logic;
    genvar i;
    generate
        for(i = 0; i < 64; i = i + 1) begin always @(posedge S_AXIS_ACLK) comp1[i] <= (comp0[i*2] > comp0[i*2 + 1] ? comp0[i*2] : comp0[i*2 + 1]); end
        for(i = 0; i < 32; i = i + 1) begin always @(posedge S_AXIS_ACLK) comp2[i] <= (comp1[i*2] > comp1[i*2 + 1] ? comp1[i*2] : comp1[i*2 + 1]); end
        for(i = 0; i < 16; i = i + 1) begin always @(posedge S_AXIS_ACLK) comp3[i] <= (comp2[i*2] > comp2[i*2 + 1] ? comp2[i*2] : comp2[i*2 + 1]); end
        for(i = 0; i < 8; i = i + 1) begin always @(posedge S_AXIS_ACLK) comp4[i] <= (comp3[i*2] > comp3[i*2 + 1] ? comp3[i*2] : comp3[i*2 + 1]); end
        for(i = 0; i < 4; i = i + 1) begin always @(posedge S_AXIS_ACLK) comp5[i] <= (comp4[i*2] > comp4[i*2 + 1] ? comp4[i*2] : comp4[i*2 + 1]); end
        for(i = 0; i < 2; i = i + 1) begin always @(posedge S_AXIS_ACLK) comp6[i] <= (comp5[i*2] > comp5[i*2 + 1] ? comp5[i*2] : comp5[i*2 + 1]); end
    endgenerate
    always @(posedge S_AXIS_ACLK) begin output_maxpool <= (comp6[0] > comp6[1] ? comp6[0] : comp6[1]); end

    // delay the input valid signal;
    generate
        for (i = 0; i < 5; i = i + 1) begin always @(posedge S_AXIS_ACLK) sr_maxpool[i + 1] <= sr_maxpool[i]; end
    endgenerate
    always @(posedge S_AXIS_ACLK) begin
        sr_maxpool[0] <= in_valid_maxpool;
        out_valid_maxpool <= sr_maxpool[5];
    end

    // Output valid of maxpool;
    reg valid_maxpool; 


    // Reserved signals for MLP;
    // Register array that holds weights and bias;
    reg [num_bits - 1:0] w1 [0:hidden_feature - 1][0:input_feature - 1]; // Regiter array holding weights of first linear layer;
    reg [num_bits - 1:0] b1 [0:hidden_feature - 1]; // Bias of first linear layer;
    reg [num_bits - 1:0] w2 [0:output_feature - 1][0:hidden_feature - 1]; // Regiter array holding weights of second linear layer;
    reg [num_bits - 1:0] b2 [0:output_feature - 1]; // Bias of second linear layer;

    // Control signals;
    reg [2:0] s_mlp; // State register in mlp processing;
    parameter t1_lnp = 3'b000; // Load a row from w1 and process;
    parameter t1_p = 3'b001; // W1 all loaded, pure processing, also write results to input;
    parameter t2_lnp = 3'b010; // Load a row from w2 and process;
    parameter t2_p = 3'b100; // W2 all loaded, pure processing

    // Control registers for mlp processing;
    reg [31:0] cnt_t1_row; // Counter of the w1 row that is being processing;
    reg [31:0] cnt_t1_result; // Counter of the result from tree 1;
    reg [31:0] cnt_t2_row; // Counter of the w2 row that is being processing;
    reg [31:0] cnt_t2_result; // Counter of the result from tree 2;



    // Inputs and outputs of the first multipler adder tree: w1(32 x 16) * x1(16 x 1) = x2(32 x 1);
    reg in_valid_t1; // Input valid signal for tree 1;
    reg out_valid_t1; // Output valid signal for tree 1;
    reg [num_bits - 1:0] input_x1 [0:input_feature - 1]; // Inputs of feature x1;
    reg [num_bits - 1:0] input_w1 [0:input_feature - 1]; // Input of weight w1;
    reg [num_bits - 1:0] output_t1; // Ouput of the adder tree, need to pass it to certain bits of x2;


    reg [num_bits - 1:0] products_t1[0:15];
    reg [num_bits - 1:0] L0sums_t1[0:7];
    reg [num_bits - 1:0] L1sums_t1[0:3];
    reg [num_bits - 1:0] L2sums_t1[0:1];
    reg [3:0] sr_valid_t1; // Adder tree 1 has a delay of 5 clk;

    // First multiply-adder tree;
    // Main arithmetic logic; 
    generate
        for (i = 0; i < 16; i = i+1) begin always @(posedge S_AXIS_ACLK) products_t1[i] <= ($signed(input_x1[i] * input_w1[i]) >>> q_bits); end
        for (i = 0; i < 8; i = i+1) begin always @(posedge S_AXIS_ACLK) L0sums_t1[i] <= products_t1[i*2] + products_t1[i*2 + 1]; end
        for (i = 0; i < 4; i = i+1) begin always @(posedge S_AXIS_ACLK) L1sums_t1[i] <= L0sums_t1[i*2] + L0sums_t1[i*2 + 1]; end
        for (i = 0; i < 2; i = i+1) begin always @(posedge S_AXIS_ACLK) L2sums_t1[i] <= L1sums_t1[i*2] + L1sums_t1[i*2 + 1]; end
    endgenerate
    always @(posedge S_AXIS_ACLK) output_t1 <= L2sums_t1[0] + L2sums_t1[1];
    // By pass valid singal with shift register;
    generate
        for (i = 0; i < 3; i = i + 1) begin always @(posedge S_AXIS_ACLK) sr_valid_t1[i + 1] <= sr_valid_t1[i]; end
    endgenerate
    always @(posedge S_AXIS_ACLK) begin
        sr_valid_t1[0] <= in_valid_t1;
        out_valid_t1 <= sr_valid_t1[3];
    end


    // Inputs and outputs of the second multipiler adder tree: w2(3 x 32) * x2(32 x 1) = result(3 x 1);
    reg in_valid_t2; // Input valid signal for tree 2;
    reg out_valid_t2; // Output valid signal for tree 2; 
    reg [num_bits - 1:0] input_x2 [0:hidden_feature - 1]; // Inputs of feature x2;
    reg [num_bits - 1:0] input_w2 [0:hidden_feature - 1]; // Input of weight w2;
    reg [num_bits - 1:0] output_t2; // Ouput of the adder tree, need to pass it to certain bits of result;


    reg [num_bits - 1:0] products_t2[0:31];
    reg [num_bits - 1:0] L0sums_t2[0:15];
    reg [num_bits - 1:0] L1sums_t2[0:7];
    reg [num_bits - 1:0] L2sums_t2[0:3];
    reg [num_bits - 1:0] L3sums_t2[0:1];
    reg [4:0] sr_valid_t2; // Adder tree 2 has a delay of 6 clk;
    
    // Second multiplier-adder tree;
    generate
        for (i = 0; i < 32; i = i+1) begin always @(posedge S_AXIS_ACLK) products_t2[i] <= ($signed(input_x2[i] * input_w2[i]) >>> q_bits); end
        for (i = 0; i < 16; i = i+1) begin always @(posedge S_AXIS_ACLK) L0sums_t2[i] <= products_t2[i*2] + products_t2[i*2 + 1]; end
        for (i = 0; i < 8; i = i+1) begin always @(posedge S_AXIS_ACLK) L1sums_t2[i] <= L0sums_t2[i*2] + L0sums_t2[i*2 + 1]; end
        for (i = 0; i < 4; i = i+1) begin always @(posedge S_AXIS_ACLK) L2sums_t2[i] <= L1sums_t2[i*2] + L1sums_t2[i*2 + 1]; end
        for (i = 0; i < 2; i = i+1) begin always @(posedge S_AXIS_ACLK) L3sums_t2[i] <= L2sums_t2[i*2] + L2sums_t2[i*2 + 1]; end
    endgenerate
    always @(posedge S_AXIS_ACLK) output_t2 <= L3sums_t2[0] + L3sums_t2[1];
    // By pass valid singal with shift register
    generate
        for (i = 0; i < 4; i = i + 1) begin always @(posedge S_AXIS_ACLK) sr_valid_t2[i + 1] <= sr_valid_t2[i]; end
    endgenerate
    always @(posedge S_AXIS_ACLK) begin
        sr_valid_t2[0] <= in_valid_t2;
        out_valid_t2 <= sr_valid_t2[4];
    end


    // Outputs from mlp;
    reg valid_mlp;


    // Reserve signals for DMA data read and write back;
    // Change the value written to result array and the output fearture size to debug;
    reg [num_bits - 1:0] result [0:output_feature - 1];
    reg [31:0] RX_count_row;
    reg [31:0] RX_count_col;
    reg [31:0] TX_count_row;
    reg [31:0] TX_count_col;
    reg [31:0] TX_count;



    // Main logic starts from here;
    //
    //
    // Configuration of controll signals;
    wire RX, RX_last, TX, TX_last;
    assign RX = S_AXIS_TVALID && S_AXIS_TREADY; // Read from DRAM;
    assign RX_last = RX && S_AXIS_TLAST; // Flag of the last transition from DRAM;
    assign TX = M_AXIS_TVALID && M_AXIS_TREADY; // Write to DRAM;
    assign TX_last = TX && M_AXIS_TLAST; // Flag of the last transition to DRAM;
    
    assign S_AXIS_TREADY = M_AXIS_TREADY; // Classifier is always ready to recieve data unless next stage is not;

    assign M_AXIS_TVALID = state == datawb || state == rd_maxpool || state == rd_x2; // Raise vaild to high in write back or debug state;
    assign M_AXIS_TLAST = (state == datawb && TX_count == output_feature - 1) ||
                          (state == rd_maxpool && TX_count == input_feature - 1) ||
                          (state == rd_x2 && TX_count == hidden_feature - 1);
    assign M_AXIS_TDATA = state == datawb ? result[TX_count] :
                          state == rd_maxpool ? input_x1[TX_count] :
                          state == rd_x2 ? input_x2[TX_count] :
                          'd0; // Output assignment;

    // State transfer rule of the FSM;q
    always @(posedge S_AXIS_ACLK) begin
        // Reset signal;
        if (~S_AXIS_ARESETN) begin
            state <= ldw1;
        end
        else begin
            // Initial state, when finished reading w1 transfer to loading b1;
            if (state == ldw1 && RX_last) begin
                // Finished reading 21;
                state <= ldb1;
            end
            // Finished reading b2, transfer to waiting for w2;
            if (state == ldb1 && RX_last) begin
                // Finished reading 21;
                state <= ldw2;
            end
            // Finished reading w2, transfer to waiting for b2;
            if (state == ldw2 && RX_last) begin
                // Finished reading 21;
                state <= ldb2;
            end
            // Finished reading b2, transfer to waiting for input data;
            if (state == ldb2 && RX_last) begin
                // Finished reading 21;
                state <= ldin;
            end
            // Finished reading input data, transfer to fft processing;
            if (state == ldin && valid_maxpool) begin
                state <= mlp;
            end
            // Finished mlp, transfer to data write back;
            if (state == mlp && valid_mlp) begin
                // state <= datawb;
                state <= rd_maxpool; // Used for debug;
            end
            // Finished data write back, return to idle and wait for input;
            if (state == datawb && TX_last) begin
                state <= ldin;
            end
            // Read input_x1;
            if (state == rd_maxpool && TX_last) begin
                state <= rd_x2;
            end
            // Read input_x2;
            if (state == rd_x2 && TX_last) begin
                state <= datawb;
            end

        end
    end

    
    // State actions of the FSM;
    always @(posedge S_AXIS_ACLK) begin
        // Global reset;
        if (~S_AXIS_ARESETN) begin
            RX_count_col <= 'd0;
            RX_count_row <= 'd0;
            TX_count_col <= 'd0;
            TX_count_row <= 'd0;
            TX_count <= 'd0;
        end
        else begin
            if ((state == ldw1 && ~RX)) begin
                RX_count_col <= 'd0;
                RX_count_row <= 'd0;
                TX_count_col <= 'd0;
                TX_count_row <= 'd0;
                TX_count <= 'd0;
            end
            // Read w1;
            if (state == ldw1 && RX) begin
                if (S_AXIS_TLAST) begin
                    RX_count_row <= 'd0;
                    RX_count_col <= 'd0;
                end 
                else begin
                    if (RX_count_col == input_feature - 1) begin
                        RX_count_col <= 'd0;
                        RX_count_row <= RX_count_row + 'd1;
                    end
                    else RX_count_col <= RX_count_col + 'd1;
                end
                w1[RX_count_row][RX_count_col] <= S_AXIS_TDATA;
            end
            // Read b1;
            if (state == ldb1 && RX) begin
                if (S_AXIS_TLAST) RX_count_col <= 'd0;
                else RX_count_col <= RX_count_col + 'd1;
                b1[RX_count_col] <= S_AXIS_TDATA;
            end
            // Read w2;
            if (state == ldw2 && RX) begin
                if (S_AXIS_TLAST) begin
                    RX_count_row <= 'd0;
                    RX_count_col <= 'd0;
                end 
                else begin
                    if (RX_count_col == hidden_feature - 1) begin
                        RX_count_col <= 'd0;
                        RX_count_row <= RX_count_row + 'd1;
                    end
                    else RX_count_col <= RX_count_col + 'd1;
                end
                w2[RX_count_row][RX_count_col] <= S_AXIS_TDATA;
            end
            // Read b2;
            if (state == ldb2 && RX) begin
                if (S_AXIS_TLAST) RX_count_col <= 'd0;
                else RX_count_col <= RX_count_col + 'd1;
                b2[RX_count_col] <= S_AXIS_TDATA;
            end
            // Read input;
            if (state == ldin) begin
                // Read samples from AXI bus with RX signal;
                if (RX) begin
                    input_fft <= S_AXIS_TDATA;
                    in_valid_fft <= 'b1;
                end
                // Set out_valid_fft to high;
                if (out_valid_fft_flag) out_valid_fft <= 'b1;
                // Readout the fft data and send it to maxpool buffer;
                if (out_valid_fft_flag | out_valid_fft) begin
                    // Read the real part of only half of the results points;
                    if (cnt_fft_point < fft_point) begin
                        comp0[cnt_fft_point] <= output_fft[41] == 1'b1 ? {9'd0, -output_fft[41:21], 2'b00}: {9'd0, output_fft[41:21], 2'b00};
                    end
                    // End of reading fft output;
                    if (cnt_fft_frame == num_frame - 1 && cnt_fft_point == fft_point_full - 1) begin
                        cnt_fft_frame <= 'd0;
                        cnt_fft_point <= 'd0;
                        // Reset the enable signal;
                        in_valid_fft <= 'b0;
                        out_valid_fft <= 'b0;
                    end
                    else begin
                        if (cnt_fft_point == fft_point_full - 1) begin
                            cnt_fft_frame <= cnt_fft_frame + 'd1;
                            cnt_fft_point <= 'd0;
                        end
                        else cnt_fft_point <= cnt_fft_point + 'd1;
                    end
                end
                // Enable max pooling of the current frame;
                if (cnt_fft_point == fft_point - 1) begin
                    in_valid_maxpool <= 'b1;
                end
                else in_valid_maxpool <= 'b0;
                // Read the output of max pooling module and write the value to mlp input;
                if (out_valid_maxpool) begin
                    input_x1[cnt_comp_result] <= output_maxpool;
                    // Reset the counter when all frame results have been read;
                    if (cnt_comp_result == num_frame - 1) begin
                        cnt_comp_result <= 'd0;
                        valid_maxpool <= 'd1;
                    end
                    else cnt_comp_result <= cnt_comp_result + 'd1;
                end
                // Reset valid signal;
                if (valid_maxpool) valid_maxpool <= 'b0;
            end
            // MLP processing;
            if (state == mlp) begin
                // Read rows of w1 and send to tree 1;
                if (s_mlp == t1_lnp) begin
                    // Increase the row counter and reset if all rows have been read;
                    if (cnt_t1_row == hidden_feature - 1) begin
                        cnt_t1_row <= 'd0;
                        s_mlp <= t1_p;
                    end 
                    else begin 
                        cnt_t1_row <= cnt_t1_row + 'd1;
                    end
                    // Load data to tree 1;
                    for (k = 0; k < input_feature; k = k + 1) begin
                        input_w1[k] <= w1[cnt_t1_row][k];
                    end
                    in_valid_t1 <= 'b1; 
                    // Write results to w2 and increase the result counter, no need to reset;
                    if (out_valid_t1) begin
                        // RELU activation;
                        input_x2[cnt_t1_result] <= $signed(output_t1 + b1[cnt_t1_result]) > 0 ? (output_t1 + b1[cnt_t1_result]) : 'd0;
                        cnt_t1_result <= cnt_t1_result + 'd1;
                    end
                end
                // Wait for tree 1 to finish processing;
                if (s_mlp == t1_p) begin
                    // No longer need to load data;
                    in_valid_t1 <= 'b0;
                    // Write results to w2 and increase the result counter, no need to reset;
                    if (out_valid_t1) begin
                        // RELU activation;
                        input_x2[cnt_t1_result] <= $signed(output_t1 + b1[cnt_t1_result]) > 0 ? (output_t1 + b1[cnt_t1_result]) : 'd0;
                        // Monitor the counter value and reset it, also transfer the state to t2_lnp;
                        if (cnt_t1_result == hidden_feature - 1) begin
                            cnt_t1_result <= 'd0;
                            s_mlp <= t2_lnp;
                        end
                        else cnt_t1_result <= cnt_t1_result + 'd1;
                    end
                end
                // Read rows of w2 and send to tree 2;
                if (s_mlp == t2_lnp) begin
                    // Increase the row counter and reset if all rows have been read;
                    if (cnt_t2_row == output_feature - 1) begin
                        cnt_t2_row <= 'd0;
                        s_mlp <= t2_p; 
                    end 
                    else begin 
                        cnt_t2_row <= cnt_t2_row + 'd1;
                    end
                    // Load data to tree 2;
                    for (k = 0; k < hidden_feature; k = k + 1) begin
                        input_w2[k] <= w2[cnt_t2_row][k];
                    end
                    in_valid_t2 <= 'b1;
                    // Write results to w2 and increase the result counter, no need to reset;
                    if (out_valid_t2) begin
                        result[cnt_t2_result] <= output_t2 + b2[cnt_t2_result];
                        cnt_t2_result <= cnt_t2_result + 'd1;
                    end
                end
                // Wait for tree 2 to finish processing;
                if (s_mlp == t2_p) begin
                    // No longer need to load data;
                    in_valid_t2 <= 'b0;
                    // Write results to w2 and increase the result counter, no need to reset;
                    if (out_valid_t2) begin
                        result[cnt_t2_result] <= output_t2 + b2[cnt_t2_result];
                        // Monitor the counter value and reset it, also transfer the state to t2_lnp;
                        if (cnt_t2_result == output_feature - 1) begin
                            cnt_t2_result <= 'd0;
                            valid_mlp <= 'b1;
                        end
                        else cnt_t2_result <= cnt_t2_result + 'd1;
                    end
                    // Reset valid_mlp and return to t1_lnp;
                    if (valid_mlp) begin
                        valid_mlp <= 'b0;
                        s_mlp <= t1_lnp;
                    end
                end
            end
            // Data write back;
            if (state == datawb && TX) begin
                if (TX_count == output_feature - 1) TX_count <= 'd0;
                else TX_count <= TX_count + 'd1;
            end

            // Debug states codes start from here;
            //
            //
            // Read input_x1;
            if (state == rd_maxpool && TX) begin
                if (TX_count == input_feature - 1) TX_count <= 'd0;
                else TX_count <= TX_count + 'd1;
            end
            // Read input_x2;
            if (state == rd_x2 && TX) begin
                if (TX_count == hidden_feature - 1) TX_count <= 'd0;
                else TX_count <= TX_count + 'd1;
            end
        end
    end

    // FFT module defition;
    fftmain FFT(
        .i_clk(S_AXIS_ACLK),
        .i_reset(~S_AXIS_ARESETN),
        .i_ce(in_valid_fft),
        .i_sample({input_fft[15:0], 16'h0000}), 
        .o_result(output_fft), 
        .o_sync(out_valid_fft_flag)
    );

    // MLP definition;



    
endmodule