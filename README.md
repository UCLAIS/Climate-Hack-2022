# [Climate-Hack-2022](https://climatehack.ai)

Climate Hack Discord: https://discord.gg/EUhf4PYpHD  
UCL AI Society Discord: https://discord.gg/GH66uWNHVX


### Description

The purpose of this repository is to present 3 possible solutions for the 'Climate Hack.AI' satellite imagery prediction challenge.

1. **Optical flow**. The first solution focuses on applying dense optical flow algorithm for future image prediction. It is based on calculating the weigthed average of the dense optical flow and using it to warp future images. Such approach does not require much computational power and provide sufficient accuracy.
2. **Conv3D**. The second solution looks how Conv3D layers can be used to capture the spatial data change in time. The project structure is similar to that of the initial example submission file found in the ClimateHack.AI [getting started guide](https://climatehack.ai/compete#2).
3. **ConvLSTM**. The last system is coded in TensorFlow rather than PyTorch and uses multiple ConvLSTM layers in addition to previously analyzed Conv3D layer. In contrast to the previous solutions, such system does not use MS-SSIM loss and instead trains with MSE loss.

Sample data file (used in training): [link](https://liveuclac-my.sharepoint.com/:u:/g/personal/zcempoc_ucl_ac_uk/Ed2SWmQrMa5JpxZDgo0JPIEBo8HAGQz5zfoWFXdrhEH26Q?e=H4jRMd)
