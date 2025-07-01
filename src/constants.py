from enum import auto, IntEnum, unique

# The maximum number of iterations that VTR will have attempted to route for. A value of 1000 means VTR
# would try to route for 1000 iterations then stop before trying the 1001st iteration.
# This would yield 1000 feature vectors HOWEVER
# There is no prediction task on the 1000th iteration, so we use VTR_MAX_ITERS-1 of the feature vectors
VTR_MAX_ITERS = 1000

# The average seconds per iteration across all Koios circuits implemented on Agilex and Flagship architectures
# Based on 2 above + 2 below Minimum channel width for classification data
AVG_SEC_PER_ITER = 10.17


class TimeIdx(IntEnum):
    SEG_WLPA = 0
    STATIC_NET = auto()
    SINK_OCC = auto()
    OVER = auto()
    SB = auto()
    WLPA = auto()
    NCPR = auto()
    ITER_PURE_ROUTE_TIME = auto()
    AGG_WALL_CLK = auto()  # This is program runtime, not routing time


class LabelIdx(IntEnum):
    NODE_TRAV = 0
    NODE_EVAL = auto()
    HEAP_PUSH = auto()
    HEAP_POP = auto()
    CONN_ROUTE = auto()
    NET_ROUTE = auto()
    ROUTE_ITER = auto()
    TOT_PURE_ROUTE_TIME = auto()


TINY_CIRCUITS = (  # Too small to be considered
    'and_latch', 'modify_count', 'mips_16', 'd_flip_flop', 'single_ff', 'single_wire', 'button_timer',
    'multiclock_reader_writer'
)

AGILEX_CIRCUITS = ('64-16bit-fixed-JACM', '8051', 'LU8PEEng', 'LU32PEEng', 'LU64PEEng', 'Md5Core', 'PWM',
                   'VexRiscvFull', 'VexRiscvFullNoMmu', 'VexRiscvFullNoMmuMaxPerf', 'VexRiscvFullNoMmuNoCache',
                   'VexRiscvLinuxBalanced', 'VexRiscvLinuxBalancedSmp', 'VexRiscvNoCacheNoMmuMaxPerf', 'VexRiscvSecure',
                   'VexRiscvSmallAndProductive', 'VexRiscvSmallAndProductiveICache', 'VexRiscvSmallest',
                   'VexRiscvSmallestNoCsr', 'aes_cipher', 'aes_inv_cipher', 'and_latch', 'arm_core', 'attention_layer',
                   'bgm', 'blob_merge', 'bnn', 'boundtop', 'button_display_control', 'button_timer',
                   'bwave_like.fixed.large', 'bwave_like.fixed.small', 'bwave_like.float.large',
                   'bwave_like.float.small', 'ch_intrinsics', 'clstm_like.large', 'clstm_like.medium',
                   'clstm_like.small', 'conv_layer', 'conv_layer_hls', 'cordic', 'd_flip_flop', 'deepfreeze.style1',
                   'deepfreeze.style2', 'deepfreeze.style3', 'diffeq1', 'diffeq2', 'dla_like.large', 'dla_like.medium',
                   'dla_like.small', 'dnnweaver', 'eltwise_layer', 'enet_core', 'ethmac', 'fpu_soft_bfly',
                   'fpu_soft_bgm', 'fpu_soft_dscg', 'fpu_soft_fir', 'fpu_soft_mm3', 'fpu_soft_ode', 'fpu_soft_syn2',
                   'fpu_soft_syn7', 'fx68kAlu', 'gemm_layer', 'lenet', 'lstm', 'mcml', 'mips_16', 'mkDelayWorker32B',
                   'mkPktMerge', 'mkSMAdapter4B', 'mmc_core', 'modify_count', 'mult_4x4', 'mult_5x5', 'mult_6x6',
                   'mult_7x7', 'mult_8x8', 'mult_9x9', 'multiclock_output_and_latch', 'multiclock_reader_writer',
                   'or1200', 'pipelined_fft_64', 'proxy.1', 'proxy.5', 'proxy.7', 'proxy.8', 'raygentop',
                   'reduction_layer', 'robot_rl', 'sha', 'single_ff', 'single_wire', 'soc_core', 'softmax', 'spmv',
                   'spree', 'stereovision0', 'stereovision1', 'stereovision2', 'stereovision3', 'tdarknet_like.large',
                   'tdarknet_like.small', 'test', 'time_counter', 'timer_display_control', 'tmds_channel',
                   'tpu_like.large.os', 'tpu_like.large.ws', 'tpu_like.small.os', 'tpu_like.small.ws', 'uaddrPla',
                   'uriscv_core', 'usb_uart_core', 'xtea')

FLAGSHIP_CIRCUITS = ('64-16bit-fixed-JACM', '8051', 'LU8PEEng', 'LU32PEEng', 'LU64PEEng', 'Md5Core', 'PWM',
                     'VexRiscvFull', 'VexRiscvFullNoMmu', 'VexRiscvFullNoMmuMaxPerf', 'VexRiscvFullNoMmuNoCache',
                     'VexRiscvLinuxBalanced', 'VexRiscvLinuxBalancedSmp', 'VexRiscvNoCacheNoMmuMaxPerf',
                     'VexRiscvSecure', 'VexRiscvSmallAndProductive', 'VexRiscvSmallAndProductiveICache',
                     'VexRiscvSmallest', 'VexRiscvSmallestNoCsr', 'aes_cipher', 'aes_inv_cipher', 'and_latch',
                     'arm_core', 'attention_layer', 'bgm', 'blob_merge', 'bnn', 'boundtop', 'button_display_control',
                     'button_timer', 'bwave_like.fixed.large', 'bwave_like.fixed.small', 'bwave_like.float.large',
                     'bwave_like.float.small', 'ch_intrinsics', 'clstm_like.large', 'clstm_like.medium',
                     'clstm_like.small', 'conv_layer', 'conv_layer_hls', 'cordic', 'd_flip_flop', 'deepfreeze.style1',
                     'deepfreeze.style2', 'deepfreeze.style3', 'diffeq1', 'diffeq2', 'dla_like.large',
                     'dla_like.medium', 'dla_like.small', 'dnnweaver', 'eltwise_layer', 'enet_core', 'ethmac',
                     'fpu_soft_bfly', 'fpu_soft_bgm', 'fpu_soft_dscg', 'fpu_soft_fir', 'fpu_soft_mm3', 'fpu_soft_ode',
                     'fpu_soft_syn2', 'fpu_soft_syn7', 'fx68kAlu', 'gemm_layer', 'lenet', 'lstm', 'mcml', 'mips_16',
                     'mkDelayWorker32B', 'mkPktMerge', 'mkSMAdapter4B', 'mmc_core', 'modify_count', 'mult_4x4',
                     'mult_5x5', 'mult_6x6', 'mult_7x7', 'mult_8x8', 'mult_9x9', 'multiclock_output_and_latch',
                     'multiclock_reader_writer', 'or1200', 'pipelined_fft_64', 'proxy.1', 'proxy.5', 'proxy.7',
                     'proxy.8', 'raygentop', 'reduction_layer', 'robot_rl', 'sha', 'single_ff', 'single_wire',
                     'soc_core', 'softmax', 'spmv', 'spree', 'stereovision0', 'stereovision1', 'stereovision2',
                     'stereovision3', 'tdarknet_like.large', 'tdarknet_like.small', 'test', 'time_counter',
                     'timer_display_control', 'tmds_channel', 'tpu_like.large.os', 'tpu_like.large.ws',
                     'tpu_like.small.os', 'tpu_like.small.ws', 'uaddrPla', 'uriscv_core', 'usb_uart_core', 'xtea')

STRATIX10_CIRCUITS = ('CH_DFSIN_stratix10_arch_timing', 'EKF-SLAM_Jacobians_stratix10_arch_timing',
                      'JPEG_stratix10_arch_timing', 'LU230_stratix10_arch_timing', 'LU_Network_stratix10_arch_timing',
                      'MCML_stratix10_arch_timing', 'MMM_stratix10_arch_timing', 'Reed_Solomon_stratix10_arch_timing',
                      'SLAM_spheric_stratix10_arch_timing', 'SURF_desc_stratix10_arch_timing',
                      'bitcoin_miner_stratix10_arch_timing', 'bitonic_mesh_stratix10_arch_timing',
                      'carpat_stratix10_arch_timing', 'cholesky_bdti_stratix10_arch_timing',
                      'cholesky_mc_stratix10_arch_timing', 'dart_stratix10_arch_timing',
                      'denoise_stratix10_arch_timing', 'des90_stratix10_arch_timing',
                      'fir_cascade_stratix10_arch_timing', 'gsm_switch_stratix10_arch_timing',
                      'jacobi_stratix10_arch_timing', 'leon2_stratix10_arch_timing', 'leon3mp_stratix10_arch_timing',
                      'mes_noc_stratix10_arch_timing', 'minres_stratix10_arch_timing', 'murax_stratix10_arch_timing',
                      'neuron_stratix10_arch_timing', 'openCV_stratix10_arch_timing', 'picosoc_stratix10_arch_timing',
                      'radar20_stratix10_arch_timing', 'random_stratix10_arch_timing',
                      'segmentation_stratix10_arch_timing', 'smithwaterman_stratix10_arch_timing',
                      'sparcT1_core_stratix10_arch_timing', 'sparcT2_core_stratix10_arch_timing',
                      'stap_qrd_stratix10_arch_timing', 'stap_steering_stratix10_arch_timing',
                      'stereo_vision_stratix10_arch_timing', 'sudoku_check_stratix10_arch_timing',
                      'ucsb_152_tap_fir_stratix10_arch_timing', 'uoft_raytracer_stratix10_arch_timing')

STRATIXIV_CIRCUITS = ('CHERI_stratixiv_arch_timing', 'CH_DFSIN_stratixiv_arch_timing',
                      'EKF-SLAM_Jacobians_stratixiv_arch_timing', 'JPEG_stratixiv_arch_timing',
                      'LU230_stratixiv_arch_timing', 'LU_Network_stratixiv_arch_timing', 'MCML_stratixiv_arch_timing',
                      'MMM_stratixiv_arch_timing', 'Reed_Solomon_stratixiv_arch_timing',
                      'SLAM_spheric_stratixiv_arch_timing', 'SURF_desc_stratixiv_arch_timing',
                      'bitcoin_miner_stratixiv_arch_timing', 'bitonic_mesh_stratixiv_arch_timing',
                      'carpat_stratixiv_arch_timing', 'cholesky_bdti_stratixiv_arch_timing',
                      'cholesky_mc_stratixiv_arch_timing', 'dart_stratixiv_arch_timing',
                      'denoise_stratixiv_arch_timing', 'des90_stratixiv_arch_timing', 'directrf_stratixiv_arch_timing',
                      'fir_cascade_stratixiv_arch_timing', 'gaussianblur_stratixiv_arch_timing',
                      'gsm_switch_stratixiv_arch_timing', 'jacobi_stratixiv_arch_timing', 'leon2_stratixiv_arch_timing',
                      'leon3mp_stratixiv_arch_timing', 'mes_noc_stratixiv_arch_timing', 'minres_stratixiv_arch_timing',
                      'murax_stratixiv_arch_timing', 'neuron_stratixiv_arch_timing', 'openCV_stratixiv_arch_timing',
                      'picosoc_stratixiv_arch_timing', 'radar20_stratixiv_arch_timing', 'random_stratixiv_arch_timing',
                      'segmentation_stratixiv_arch_timing', 'smithwaterman_stratixiv_arch_timing',
                      'sparcT1_chip2_stratixiv_arch_timing', 'sparcT1_core_stratixiv_arch_timing',
                      'sparcT2_core_stratixiv_arch_timing', 'stap_qrd_stratixiv_arch_timing',
                      'stap_steering_stratixiv_arch_timing', 'stereo_vision_stratixiv_arch_timing',
                      'sudoku_check_stratixiv_arch_timing', 'ucsb_152_tap_fir_stratixiv_arch_timing',
                      'uoft_raytracer_stratixiv_arch_timing', 'wb_conmax_stratixiv_arch_timing')
