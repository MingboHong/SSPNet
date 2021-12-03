import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.Module.CBAM.CBAM import CBAM_block
from mmcv.cnn import normal_init
from ..Highlight_Attention.Attention import SSPAttBlock, SCAttBlock


class MS_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs=5):
        super(MS_CBAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.ms_cbam = CBAM_block(in_planes=self.in_channels, out_planes=self.out_channels)

        self.connect_op = nn.ModuleList()
        for i in range(self.num_outs - 1):
            self.connect_op.append(
                nn.Conv2d(self.num_outs - i, 1, kernel_size=3, padding=1, stride=1)
            )

        self.sigmoid = nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        """
        MS_CBAM combines the other attention map

        For example:
        for c-3 feature map
        STEP1:
        >>> (3, 256, 64, 64) ->CBAM_block ->(3, 1, 64, 64)
        >>> (3, 256, 32, 32) ->CBAM_block ->(3, 1, 32, 32)   ->Upsample_2x ->(3, 1, 64, 64)
        >>> (3, 256, 16, 16) ->CBAM_block ->(3, 1, 16, 16)   ->Upsample_4x ->(3, 1, 64, 64)
        >>> (3, 256, 8, 8)   ->CBAM_block ->(3, 1, 8, 8)     ->Upsample_8x ->(3, 1, 64, 64)
        >>> (3, 256, 4, 4)   ->CBAM_block ->(3, 1, 4, 4)     ->Upsample_16x ->(3, 1, 64, 64)
        STEP2:
        An connect opreation was utilized to fuse mutili-scale attention map
        >>>[(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64)]

        >>>(3, 5, 64, 64)  ->connect_op  -> (3, 1, 64, 64)

        Args:
            inputs: output of FPN module

        Returns:

        """
        assert len(inputs) == self.num_outs
        out = []
        pyrmaid_feature_list = []

        for i in range(self.num_outs):
            tmp_out = self.ms_cbam(inputs[i])
            out.append(tmp_out)

        for i in range(self.num_outs - 1):
            tmp_pyrmaid = []
            tmp_pyrmaid.append(out[i])
            for up, j in enumerate(range(i + 1, self.num_outs)):
                tmp_out = F.interpolate(out[j], scale_factor=2**(up+1), mode='bilinear')
                tmp_pyrmaid.append(tmp_out)
            pyrmaid_feature = torch.cat(tmp_pyrmaid, dim=1)
            #pyrmaid_feature = self.pad(pyrmaid_feature)
            att_map = self.connect_op[i](pyrmaid_feature)
            att_map = self.sigmoid(att_map)

            pyrmaid_feature_list.append(att_map)

        pyrmaid_feature_list.append(out[-1])
        output = [inputs[i] * pyrmaid_feature_list[i] + inputs[i] for i in range(self.num_outs)]


        return output


if __name__ == '__main__':
    sc_1 = torch.randn((3, 256, 240, 192))
    sc_2 = torch.randn((3, 256, 120, 96))
    sc_3 = torch.randn((3, 256, 60, 48))
    sc_4 = torch.randn((3, 256, 30, 24))
    sc_5 = torch.randn((3, 256, 15, 12))
    sc_ms = tuple((sc_1, sc_2, sc_3, sc_4, sc_5))
    att_block = MS_CBAM(256, 256)
    att_map = att_block(sc_ms)

    #    b, c, h, w = pyrmaid_feature_list[0].size()
    # att_map = self.softmax((pyrmaid_feature_list[0]).view(b, -1)).view(b, c, h, w)
    #    att = (pyrmaid_feature_list[0].cpu().numpy()[0])*255*5
    #    att = att.astype(np.uint8).transpose(1, 2, 0)
    #    att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    # mmcv.imshow(att)
    # cv2.waitKey(0)
"""
    def forward(self, inputs):
    
        MS_CBAM combines the other attention map

        For example:
        for c-3 feature map
        STEP1:
        >>> (3, 256, 64, 64) ->CBAM_block ->(3, 1, 64, 64)
        >>> (3, 256, 32, 32) ->CBAM_block ->(3, 1, 32, 32)   ->Upsample_2x ->(3, 1, 64, 64)
        >>> (3, 256, 16, 16) ->CBAM_block ->(3, 1, 16, 16)   ->Upsample_4x ->(3, 1, 64, 64)
        >>> (3, 256, 8, 8)   ->CBAM_block ->(3, 1, 8, 8)     ->Upsample_8x ->(3, 1, 64, 64)
        >>> (3, 256, 4, 4)   ->CBAM_block ->(3, 1, 4, 4)     ->Upsample_16x ->(3, 1, 64, 64)
        STEP2:
        An connect opreation was utilized to fuse mutili-scale attention map
        >>>[(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64)]

        >>>(3, 5, 64, 64)  ->connect_op  -> (3, 1, 64, 64)

        Args:
            inputs: output of FPN module

        Returns:

        
        assert len(inputs) == self.num_outs
        out = []
        pyrmaid_feature_list = []
        for i in range(self.num_outs):
            tmp_out = self.ms_cbam(inputs[i])
            out.append(tmp_out)

        for i in range(self.num_outs):
            tmp_pyrmaid = []
            if i != 0:
                for up_i in range(i):
                    tmp_out = self.pooling(out[up_i], kernel_size=2 ** (i - up_i), stride=2 ** (i - up_i))
                    tmp_pyrmaid.append(tmp_out)
            tmp_pyrmaid.append(out[i])
            if i != self.num_outs - 1:
                for down_i, idx in enumerate(range(i + 1, self.num_outs)):
                    tmp_out = F.interpolate(out[idx], scale_factor=2 ** (down_i + 1), mode='bilinear')
                    tmp_pyrmaid.append(tmp_out)

            pyrmaid_feature = torch.cat(tmp_pyrmaid, dim=1)
            att_map = self.connect_op(pyrmaid_feature)
            b, c, h, w = att_map.size()
            att_map = self.softmax(att_map.view(b, -1)).view(b, c, h, w)
            pyrmaid_feature_list.append(att_map)
        output = [inputs[i] * pyrmaid_feature_list[i]+inputs[i] for i in range(self.num_outs)]
        return output



"""

"""
  def forward(self, inputs):
      
        MS_CBAM combines the other attention map

        For example:
        for c-3 feature map
        STEP1:
        >>> (3, 256, 64, 64) ->CBAM_block ->(3, 1, 64, 64)
        >>> (3, 256, 32, 32) ->CBAM_block ->(3, 1, 32, 32)   ->Upsample_2x ->(3, 1, 64, 64)
        >>> (3, 256, 16, 16) ->CBAM_block ->(3, 1, 16, 16)   ->Upsample_4x ->(3, 1, 64, 64)
        >>> (3, 256, 8, 8)   ->CBAM_block ->(3, 1, 8, 8)     ->Upsample_8x ->(3, 1, 64, 64)
        >>> (3, 256, 4, 4)   ->CBAM_block ->(3, 1, 4, 4)     ->Upsample_16x ->(3, 1, 64, 64)
        STEP2:
        An connect opreation was utilized to fuse mutili-scale attention map
        >>>[(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64),
        >>>(3, 1, 64, 64)]

        >>>(3, 5, 64, 64)  ->connect_op  -> (3, 1, 64, 64)

        Args:
            inputs: output of FPN module

        Returns:

   
        assert len(inputs) == self.num_outs
        out = []
        pyrmaid_feature_list = []
        for i in range(self.num_outs):
            tmp_out = self.ms_cbam(inputs[i])
            out.append(tmp_out)


        tmp_pyrmaid = []
        tmp_pyrmaid.append(out[0])

        for i in range(1, self.num_outs):
            tmp_out = F.interpolate(out[i], scale_factor=2 ** i, mode='bilinear')
            tmp_pyrmaid.append(tmp_out)

        pyrmaid_feature = torch.cat(tmp_pyrmaid, dim=1)
        att_map = self.connect_op(pyrmaid_feature)
        # b, c, h, w = att_map.size()
        # att_map = self.softmax(att_map.view(b, -1)).view(b, c, h, w)
        att_map = self.sigmoid(att_map)
        # att = (att_map.cpu().numpy()[0])*255*1000
        # att = att.astype(np.uint8).transpose(1, 2, 0)
        # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        # 
        pyrmaid_feature_list.append(att_map)
        for i in range(1, self.num_outs):
            tmp_out = self.pooling(att_map, kernel_size=2 ** i, stride=2 ** i)
            pyrmaid_feature_list.append(tmp_out)


        output = [inputs[i] * pyrmaid_feature_list[i]+inputs[i] for i in range(self.num_outs)]
        return output






"""
