from gym.envs.registration import register

register(
    id='sdnsampling-v0',
    entry_point='sdn_sampling.envs:SdnSamplingEnv',
    )

register(
    id='sdnsampling-v1',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_v1',
    )

register(
    id='sdnsampling-v2',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_v2',
    )

register(
    id='sdnsampling-v3',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_v3',
    )

register(
    id='sdnsampling-v4',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_v4',
    )

register(
    id='sdnsampling-v5',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_v5',
    )

register(
    id='sdnsampling-MTD-v0',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_v0',
    )

register(
    id='sdnsampling-MTD-v1',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_v1',
    )

register(
    id='sdnsampling-MTD-v2',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_v2',
    )

register(
    id='sdnsampling-MTD-as-v0',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_as_v0',
    )

register(
    id='sdnsampling-MTD-as-v2',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_as_v2',
    )

register(
    id='sdnsampling-exp-v0',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_exp_v0',
    )

register(
    id='sdnsampling-exp-v1',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_exp_v1',
    )

register(
    id='sdnsampling-exp-v2',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_exp_v2',
    )

register(
    id='sdnsampling-MTD-exp-v0',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_exp_v0',
    )

register(
    id='sdnsampling-MTD-exp-v1',
    entry_point='sdn_sampling.envs:SdnSamplingEnv_MTD_exp_v1',
    )