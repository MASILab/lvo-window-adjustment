import pandas as pd


def main():
    cta = '/nfs/masi/lingams/lvo/ss_betctregmask_cta_registered/' \
          'copied__0-1-balanced150each_LABELS_ct_cta_08202020.csv'

    cta_windowed = '/nfs/masi/lingams/lvo/windowmanual_ss_betctregmask_cta_registered/' \
                   'copied__0-1-balanced150each_LABELS_ct_cta_08202020.csv'

    mip = '/nfs/masi/lingams/lvo/' \
          'mip_d2s120t40_ss_betctregmask_cta_registered/' \
          'copied__0-1-balanced150each_LABELS_ct_cta_08202020.csv'

    mip_windowed = '/nfs/masi/lingams/lvo/' \
                    'mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered/' \
                    'copied__0-1-balanced150each_LABELS_ct_cta_08202020.csv'

    df_mip = pd.read_csv(mip)
    df_mip_wd = pd.read_csv(mip_windowed)

    df_cta = pd.read_csv(cta)
    df_cta_wd = pd.read_csv(cta_windowed)

    df_cta = df_cta[['subj', 'set', 'lvo_radreport', 'window_level_manual', 'window_width_manual',
                     'copied__ss_betctregmask_cta_registered']]
    df_cta_wd = df_cta_wd[['subj', 'copied__windowmanual_ss_betctregmask_cta_registered']]
    df_cta_merge = df_cta.merge(df_cta_wd, left_on='subj', right_on='subj', how='outer')

    df_mip = df_mip[['subj', 'copied__mip_d2s120t40_ss_betctregmask_cta_registered']]
    df_mip_wd = df_mip_wd[['subj', 'copied__mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered']]
    df_mip_merge = df_mip.merge(df_mip_wd, left_on='subj', right_on='subj', how='outer')

    df_combined = df_cta_merge.merge(df_mip_merge, left_on='subj', right_on='subj', how='outer')

    df_combined = df_combined.rename(columns={'copied__mip_d2s120t40_ss_betctregmask_cta_registered': 'mip',
                                              'copied__mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered': 'mip_windowed',
                                              'copied__ss_betctregmask_cta_registered': 'cta',
                                              'copied__windowmanual_ss_betctregmask_cta_registered': 'cta_windowed'})

    # add binned label to annotate whether the manual window level is less 200 or greater or equal to 200
    df_bin = df_combined.copy()
    df_bin = df_bin[(df_bin['window_level_manual'] <= 180) | (df_bin['window_level_manual'] >= 220)]
    label_list = []
    for idx, row in df_bin.iterrows():
        if row['window_level_manual'] <= 180:
            label_list.append(0)
        else:
            label_list.append(1)

    df_bin['binned_label'] = label_list
    print('binned label is 0 in train:', len(df_bin[(df_bin['binned_label'] == 0) & (df_bin['set']=='train')]),
          'binned label is 1 in train: ', len(df_bin[(df_bin['binned_label'] == 1) & (df_bin['set']=='train')]))

    print('binned label is 0 in validation:', len(df_bin[(df_bin['binned_label'] == 0) & (df_bin['set'] == 'val')]),
          'binned label is 1 in validation: ', len(df_bin[(df_bin['binned_label'] == 1) & (df_bin['set'] == 'val')]))

    # add regression value
    df_reg_level = df_combined.copy()
    df_reg_level['regression_level'] = df_reg_level['window_level_manual']

    df_reg_width = df_combined.copy()
    df_reg_width['regression_width'] = df_reg_width['window_width_manual']

    df_reg_sq = df_combined.copy()
    df_reg_sq['window_level_manual'] = df_reg_sq['window_level_manual'].apply(lambda x:x**2)

    df_reg_sq.to_csv('csv/dataset_sq_test.csv', index=0)
    df_bin.to_csv('csv/dataset_bin_level.csv', index=0)
    df_reg_level.to_csv('csv/dataset_reg_level.csv', index=0)
    df_reg_width.to_csv('csv/dataset_reg_width.csv', index=0)
    df_combined.to_csv('csv/dataset.csv', index=0)


if __name__ == '__main__':
    main()


