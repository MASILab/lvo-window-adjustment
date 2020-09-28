import pandas as pd


def main():
    mip = '/nfs/masi/lingams/lvo/' \
                   'mip_d2s120t40_ss_betctregmask_cta_registered/' \
                   'copied__0-1-balanced150each_LABELS_ct_cta_08202020.csv'

    mip_windowed = '/nfs/masi/lingams/lvo/' \
          'mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered/' \
          'copied__0-1-balanced150each_LABELS_ct_cta_08202020.csv'

    df_mip = pd.read_csv(mip)
    df_mip_wd = pd.read_csv(mip_windowed)

    df_mip = df_mip[['subj', 'set', 'lvo_radreport', 'window_level_manual', 'window_width_manual',
                     'copied__mip_d2s120t40_ss_betctregmask_cta_registered']]
    df_mip_wd = df_mip_wd[['subj', 'copied__mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered']]

    df_combined = df_mip.merge(df_mip_wd, left_on='subj', right_on='subj', how='outer')
    df_combined = df_combined.rename(columns={'copied__mip_d2s120t40_ss_betctregmask_cta_registered': 'original',
                                              'copied__mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered': 'adjusted'})
    df_combined.to_csv('dataset.csv', index=0)


if __name__ == '__main__':
    main()


