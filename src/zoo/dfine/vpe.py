import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .box_ops import box_cxcywh_to_xyxy

# -----------------------------
# Box positional encoding
# -----------------------------
class BoxPositionalEncoding_wob(nn.Module):
    def __init__(self, embed_dim=256, temperature=10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.dim_each = embed_dim // 4
        assert self.dim_each*4 == embed_dim

        self.linear = nn.Linear(self.dim_each*4, embed_dim)

    def forward(self, boxes):
        """
        boxes: [N,4] normalized [0,1] -> x_min, y_min, x_max, y_max
        returns: [N, embed_dim]
        """
        N = boxes.shape[0]
        dim_each = self.embed_dim // 4
        device = boxes.device
        
        coords = [boxes[:, i].unsqueeze(1) for i in range(4)]  # list of 4 tensors [N,1]

        # create frequency terms: [dim_each]
        dim_each = self.dim_each
        if dim_each == 0:
            # fallback: embed_dim < 4 (rare); just linear map
            out = boxes
            if self.proj is not None:
                out = self.proj(out)
            return out

        # position embedding base: [dim_each]
        # we will create even indices for sin and odd for cos, like transformer
        position = torch.arange(dim_each, dtype=torch.float32, device=device)
        # the usual formula uses temperature^(2i/dim)
        denom = torch.pow(self.temperature, (2 * (position // 2)) / float(dim_each))
        embs = []
        for coord in coords:
            # coord: [N,1], denom: [dim_each]
            scaled = coord / denom.unsqueeze(0)   # broadcasting -> [N, dim_each]
            sin_emb = torch.sin(scaled[:, 0::2])  # even dims
            cos_emb = torch.cos(scaled[:, 1::2])  # odd dims
            # interleave sin and cos to produce dim_each dims
            # handle when dim_each is odd by padding
            interleaved = torch.zeros(N, dim_each, device=device)
            interleaved[:, 0::2] = sin_emb
            if interleaved.shape[1] > 1:
                interleaved[:, 1::2] = cos_emb
            embs.append(interleaved)  # [N, dim_each]

        # concat 4 coords -> [N, dim_each*4]
        pe = torch.cat(embs, dim=1)

        # if embed_dim not divisible by 4, project to exact embed_dim
        if self.linear is not None:
            pe = self.linear(pe)

        return pe  # [N, embed_dim]


class BoxPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=256, temperature=10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = float(temperature)
        self.dim_each = embed_dim // 4
        
        self.proj = nn.Linear(self.dim_each * 4, embed_dim)

    def forward(self, boxes):
        """
        boxes: [B, Nq, 4], normalized coords in [0,1]
        returns: [B, Nq, embed_dim]
        """
        B, Nq, _ = boxes.shape
        device = boxes.device
        dim_each = self.dim_each
        if dim_each == 0:
            out = boxes
            if self.proj is not None:
                out = self.proj(out)
            return out

        # position index
        position = torch.arange(dim_each, dtype=torch.float32, device=device)  # [dim_each]
        two_i = (2.0 * (position // 2.0))
        denom = torch.pow(torch.tensor(self.temperature, device=device), two_i / float(dim_each))  # [dim_each]

        # boxes: [B,Nq,4] -> coords_expanded [B,Nq,4,dim_each]
        coords_expanded = boxes.unsqueeze(-1) / denom.view(1, 1, -1)  # broadcasting -> [B,Nq,4,dim_each]

        # prepare pe container [B,Nq,4*dim_each]
        pe = torch.zeros((B, Nq, 4 * dim_each), device=device, dtype=torch.float32)

        # compute sin/cos interleaved per coord
        # loop over 4 coords (constant small loop ok for ONNX)
        for i in range(4):
            c = coords_expanded[:, :, i, :]  # [B,Nq,dim_each]
            inter = torch.zeros_like(c)
            inter[..., 0::2] = torch.sin(c[..., 0::2])
            if dim_each > 1:
                inter[..., 1::2] = torch.cos(c[..., 1::2])
            pe[:, :, i * dim_each:(i + 1) * dim_each] = inter

        if self.proj is not None:
            # reshape to [B*Nq, 4*dim_each] -> proj -> [B*Nq, embed_dim] -> reshape back
            pe = pe.view(B * Nq, -1)
            pe = self.proj(pe)
            pe = pe.view(B, Nq, -1)

        return pe  # [B, Nq, embed_dim]


# -----------------------------
# MS Deformable Cross Attention (supports batch_size=1)
# -----------------------------
class MSDeformableCrossAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3):
        """
        embed_dim: feature dim
        n_heads: number of attention heads
        n_points: sampling points per head per level
        n_levels: number of feature levels (scales)
        """
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.head_dim = embed_dim // n_heads

        # projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        # value projection implemented as 1x1 conv per level for convenience
        self.value_projs = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(n_levels)])
        # predict offsets and attention weights:
        # offsets -> [N_query, n_heads * n_points * 2 * n_levels]
        # attn weights -> [N_query, n_heads * n_points * n_levels]
        self.sampling_offsets = nn.Linear(embed_dim, n_heads * n_points * 2 * n_levels)
        self.attn_weights = nn.Linear(embed_dim, n_heads * n_points * n_levels)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # initialization (small offsets)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attn_weights.weight, 0.)
        nn.init.constant_(self.attn_weights.bias, 0.)
        # small random init for output proj
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, feat_maps, boxes):
        """
        query: [N_query, C]
        feat_maps: list of feature maps per level, each [B=1, C, H_l, W_l]
        boxes: [N_query,4] normalized coords (x_min,y_min,x_max,y_max) in [0,1]
        returns: [N_query, C]
        NOTE: this implementation assumes B==1 (batch size 1)
        """
        assert len(feat_maps) == self.n_levels
        device = query.device
        N_query, C = query.shape
        B = feat_maps[0].shape[0]
        assert B == 1, "This implementation targets batch_size==1"

        # 1) projections
        query_proj = self.query_proj(query)  # [N_query, C]

        # 2) predict offsets and attention weights
        # sampling_offsets: [N_query, n_heads * n_points * 2 * n_levels]
        sampling_offsets = self.sampling_offsets(query_proj)
        sampling_offsets = sampling_offsets.view(N_query, self.n_heads, self.n_points, 2, self.n_levels)
        # attn_weights: [N_query, n_heads * n_points * n_levels] -> reshape and softmax over points dimension
        attn_weights = self.attn_weights(query_proj)
        attn_weights = attn_weights.view(N_query, self.n_heads, self.n_points, self.n_levels)
        # softmax over points dimension (n_points) - per head & level
        attn_weights = F.softmax(attn_weights, dim=2)  # sum over points = 1

        # 3) prepare per-level sampling
        # compute box centers and sizes
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5  # [N_query]
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5  # [N_query]
        bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)  # [N_query]
        bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)  # [N_query]

        # will accumulate per-level sampled features: list of [N_query, C]
        sampled_per_level = []

        for lvl, fm in enumerate(feat_maps):
            # fm: [1, C, H, W]
            _, _, H, W = fm.shape
            # project value
            value = self.value_projs[lvl](fm)  # [1, C, H, W]

            # get offsets for this level: [N_query, n_heads, n_points, 2]
            offsets_lvl = sampling_offsets[..., :, :, :, lvl]  # shape already [N_query, heads, points, 2]
            # interpret offsets as relative to box size (this is common): offsets in range ~[-0.5,0.5] * (box_w/box_h)
            # so we multiply offsets by box_w / box_h respectively.
            # shape expand to [N_query, heads, points, 2]
            # broadcast box dims
            bw_exp = bw.view(N_query, 1, 1, 1)  # [N_query,1,1,1]
            bh_exp = bh.view(N_query, 1, 1, 1)
            # offsets_x scaled by bw, offsets_y scaled by bh
            # (we assume offsets are in units of box width/height)
            grid_x = cx.view(N_query, 1, 1, 1) + offsets_lvl[..., 0:1] * bw_exp  # [N_query,heads,points,1]
            grid_y = cy.view(N_query, 1, 1, 1) + offsets_lvl[..., 1:2] * bh_exp  # [N_query,heads,points,1]
            # now grid_x/y are normalized coords in [0,1] (hopefully); convert to grid_sample coords [-1,1]
            grid_x = grid_x.clamp(0.0, 1.0)
            grid_y = grid_y.clamp(0.0, 1.0)
            grid_sample_x = grid_x * 2.0 - 1.0  # [-1,1]
            grid_sample_y = grid_y * 2.0 - 1.0

            # build sampling grid for grid_sample
            # desired shape: [B=1, n_pts_total, 1, 2] where n_pts_total = N_query * n_heads * n_points
            # grid_sample expects last dim = (x, y)
            n_pts_total = N_query * self.n_heads * self.n_points
            # flatten and reorder to match grid_sample calling convention
            gx = grid_sample_x.view(N_query * self.n_heads * self.n_points)  # [n_pts_total]
            gy = grid_sample_y.view(N_query * self.n_heads * self.n_points)
            grid = torch.stack((gx, gy), dim=-1).view(1, n_pts_total, 1, 2).to(device)  # [1, n_pts_total, 1, 2]

            # grid_sample on value: input [1,C,H,W], grid [1,n_pts_total,1,2] -> out [1, C, n_pts_total, 1]
            sampled = F.grid_sample(value, grid, mode='bilinear', align_corners=False, padding_mode='zeros')  # [1, C, n_pts_total, 1]
            sampled = sampled.view(1, C, n_pts_total)  # [1,C,n_pts_total]
            # reshape to [N_query, n_heads, n_points, C]
            sampled = sampled.permute(2, 0, 1).view(n_pts_total, 1, C)  # [n_pts_total,1,C]
            # Now reorganize
            sampled = sampled.view(N_query, self.n_heads, self.n_points, C)  # [N_query,heads,points,C]

            # apply attention weights (for this level): [N_query,heads,points] -> expand to match C
            attn_w_lvl = attn_weights[..., lvl]  # [N_query,heads,points]
            attn_w_lvl = attn_w_lvl.unsqueeze(-1)  # [N_query,heads,points,1]

            sampled_weighted = (sampled * attn_w_lvl).sum(dim=2)  # sum over points -> [N_query,heads,C]
            # merge heads: concat or sum? In Deformable DETR heads are merged by linear proj; here we sum heads on feature dim
            sampled_heads_merged = sampled_weighted.view(N_query, -1, C).sum(dim=1)  # [N_query,C]
            # collect
            sampled_per_level.append(sampled_heads_merged)  # list of [N_query,C]

        # 4) aggregate levels
        feat_out = sum(sampled_per_level) / float(self.n_levels)  # [N_query,C]

        # 5) final linear
        out = self.output_proj(feat_out)  # [N_query,C]
        return out

class MSDeformableCrossAttentionBatchONNX(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.head_dim = embed_dim // n_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.value_projs = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(n_levels)])
        self.sampling_offsets = nn.Linear(embed_dim, n_heads * n_points * 2 * n_levels)
        self.attn_weights = nn.Linear(embed_dim, n_heads * n_points * n_levels)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attn_weights.weight, 0.)
        nn.init.constant_(self.attn_weights.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, feat_maps, boxes, box_mask=None, padding_masks=None):
        """
        query: [B, Nq, C]
        feat_maps: list of n_levels feature maps, each [B, C, H_l, W_l]
        boxes: [B, Nq, 4] normalized coords
        box_mask: [B, Nq] (0/1), optional. If None, all boxes considered valid.
        returns: [B, Nq, C]
        """
        B, Nq, C = query.shape
        device = query.device
        assert len(feat_maps) == self.n_levels

        if box_mask is None:
            box_mask = torch.ones((B, Nq), dtype=torch.float32, device=device)
        
        # project query
        query_flat = query.view(B * Nq, C)  # [B*Nq, C]
        query_proj = self.query_proj(query_flat)  # [B*Nq, C]

        # predict offsets & weights -> reshape to per-batch, per-query dims
        sampling_offsets = self.sampling_offsets(query_proj)  # [B*Nq, n_heads*n_points*2*n_levels]
        sampling_offsets = sampling_offsets.view(B, Nq, self.n_heads, self.n_points, 2, self.n_levels)
        attn = self.attn_weights(query_proj).view(B, Nq, self.n_heads, self.n_points, self.n_levels)
        attn = F.softmax(attn, dim=3)  # softmax over points dim

        # box centers & sizes
        cx = (boxes[..., 0] + boxes[..., 2]) * 0.5  # [B,Nq]
        cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
        bw = (boxes[..., 2] - boxes[..., 0]).clamp(min=1e-6)
        bh = (boxes[..., 3] - boxes[..., 1]).clamp(min=1e-6)

        # for each level, do grid_sample across batch simultaneously
        sampled_levels = []
        for lvl_idx, fm in enumerate(feat_maps):
            # fm: [B, C, H, W]
            Bf, Cf, H, W = fm.shape
            assert Bf == B, "feature batch mismatch"
            
            #if padding_masks is None:
            #    lpadding_masks = torch.ones((Bf, H, W), device=query.device)
            
            #else:
            #    # 做缩放
            #    if len(padding_masks.size()) == 3:
            #        padding_masks = padding_masks.unsqueeze(1) # 需要[B, C, H, W]
            #    lpadding_masks = F.interpolate(padding_masks.unsqueeze(1), size=(H, W), mode='bilinear')
            
            # value projection
            value = self.value_projs[lvl_idx](fm)  # [B, C, H, W]

            # offsets for this level: [B, Nq, heads, points, 2]
            offsets_lvl = sampling_offsets[..., :, :, :, lvl_idx]  # slicing by level axis

            # scale offsets by box size -> produce absolute normalized coords
            # offsets interpreted relative to box width/height
            # offsets_lvl[...,0] * bw + cx  => normalized in [0,1]
            bw_exp = bw.view(B, Nq, 1, 1)  # [B,Nq,1,1]
            bh_exp = bh.view(B, Nq, 1, 1)
            grid_x = cx.view(B, Nq, 1, 1) + offsets_lvl[..., 0] * bw_exp  # [B,Nq,heads,points]
            grid_y = cy.view(B, Nq, 1, 1) + offsets_lvl[..., 1] * bh_exp

            # clamp to [0,1]
            grid_x = grid_x.clamp(0.0, 1.0)
            grid_y = grid_y.clamp(0.0, 1.0)

            # convert to grid_sample coords [-1,1]
            grid_x = grid_x * 2.0 - 1.0
            grid_y = grid_y * 2.0 - 1.0

            # flatten per image: we need grid shape [B, n_pts_per_image, 1, 2]
            # n_pts_per_image = Nq * heads * points
            n_pts_per_image = Nq * self.n_heads * self.n_points
            # reshape ordering: [B, Nq, heads, points] -> [B, n_pts_per_image]
            gx = grid_x.permute(0, 2, 3, 1).contiguous().view(B, n_pts_per_image)  # careful ordering to match sampled reshape later
            gy = grid_y.permute(0, 2, 3, 1).contiguous().view(B, n_pts_per_image)

            # stack to grid [B, n_pts_per_image, 1, 2], grid_sample expects last dim (x,y)
            grid = torch.stack((gx, gy), dim=-1).view(B, n_pts_per_image, 2)
            grid = grid.unsqueeze(2)  # [B, n_pts_per_image, 1, 2]

            # grid_sample on value: input [B, C, H, W], grid [B, n_pts, 1, 2] -> out [B, C, n_pts, 1]
            sampled = F.grid_sample(value, grid, mode='bilinear', align_corners=False, padding_mode='zeros')  # [B, C, n_pts, 1]
            sampled = sampled.view(B, C, n_pts_per_image)  # [B, C, n_pts]
            # want shape [B, Nq, heads, points, C]
            # current ordering of points must match how we flattened gx/gy:
            # we used permute(0,2,3,1) meaning indices: (B, heads, points, Nq) flattened -> we must unflatten accordingly
            sampled = sampled.permute(0, 2, 1)  # [B, n_pts, C]
            sampled = sampled.view(B, self.n_heads, self.n_points, Nq, C)  # [B,heads,points,Nq,C]
            sampled = sampled.permute(0, 3, 1, 2, 4).contiguous()  # [B,Nq,heads,points,C]

            # get attn weights for this level: [B,Nq,heads,points] -> expand to C
            attn_lvl = attn[..., lvl_idx]  # [B,Nq,heads,points]
            attn_lvl = attn_lvl.unsqueeze(-1)  # [B,Nq,heads,points,1]

            # weighted sum over points -> [B,Nq,heads,C]
            weighted = (sampled * attn_lvl).sum(dim=3)  # sum points

            # merge heads: here sum over heads into feature dim (simple), alternative: concat + linear
            merged = weighted.view(B, Nq, -1, C).sum(dim=2)  # [B,Nq,C]
            sampled_levels.append(merged)

        # aggregate levels and project
        feat_out = sum(sampled_levels) / float(self.n_levels)  # [B,Nq,C]
        out = self.output_proj(feat_out.view(B * Nq, C)).view(B, Nq, C)

        # optional: zero out masked boxes (so logits for padded boxes are neutral)
        # but better keep as is and use box_mask upstream to ignore them in loss
        return out  # [B, Nq, C]


class MSDeformableCrossAttentionBatch_wmask(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3, base_scale=0.5):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.head_dim = embed_dim // n_heads
        self.base_scale = base_scale  # controls initial grid half-range

        # projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        # value projection per level (1x1 conv)
        self.value_projs = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(n_levels)])
        # offsets & attn
        self.sampling_offsets = nn.Linear(embed_dim, n_heads * n_points * 2 * n_levels)
        self.attn_weights = nn.Linear(embed_dim, n_heads * n_points * n_levels)
        # after concat heads (heads * head_dim == embed_dim) -> output_proj keeps embed_dim
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters_uniform_grid_2d()

    def _reset_parameters_uniform_grid_2d(self):
        """
        Initialize sampling_offsets.bias with 2D uniform grid points in [-scale, scale] relative to box size,
        scale increases with level: scale = base_scale * (lvl + 1) / n_levels  (or another schedule).
        Different heads get small cyclic shifts to avoid perfect overlap.
        """
        n_h = self.n_heads
        n_p = self.n_points
        n_l = self.n_levels
        device = self.sampling_offsets.bias.device if hasattr(self.sampling_offsets, 'bias') else torch.device('cpu')

        # Determine grid shape (rows x cols) as near-square as possible
        grid_rows = int(math.floor(math.sqrt(n_p)))
        grid_cols = int(math.ceil(n_p / grid_rows))
        # build base coords in [-1,1] grid positions centered
        xs = torch.linspace(-0.5, 0.5, steps=grid_cols, dtype=torch.float32, device=device)
        ys = torch.linspace(-0.5, 0.5, steps=grid_rows, dtype=torch.float32, device=device)
        base_pts = []
        for yi in ys:
            for xi in xs:
                base_pts.append((xi.item(), yi.item()))
        # trim to n_p if overshot
        base_pts = base_pts[:n_p]  # list of (x,y), length n_p

        # Allocate bias array: shape [n_h, n_p, 2, n_l]
        bias = torch.zeros((n_h, n_p, 2, n_l), dtype=torch.float32, device=device)

        for lvl in range(n_l):
            # choose scale per level; level 0 small, higher level larger
            # you can tune schedule; here scale grows linearly but capped to base_scale*(lvl+1)
            # we map base_scale so that points lie roughly in [-scale, scale] relative to box size
            scale = self.base_scale * (1.0 + lvl / max(1, n_l - 1))  # from base_scale to base_scale*2
            for h in range(n_h):
                # head-specific phase shift (small) to diversify sampling
                phase = (h / n_h) * (1.0 / max(1, n_p))
                for p_idx, (bx, by) in enumerate(base_pts):
                    # apply small cyclic shift depending on head and point index
                    shift_x = phase * ( (p_idx % n_p) / n_p )
                    shift_y = -phase * ( (p_idx % n_p) / n_p )
                    x = bx * scale + shift_x
                    y = by * scale + shift_y
                    bias[h, p_idx, 0, lvl] = x
                    bias[h, p_idx, 1, lvl] = y

        # flatten bias to linear bias shape
        bias_flat = bias.reshape(-1)  # length = n_h * n_p * 2 * n_l
        with torch.no_grad():
            # set linear bias and small init for weights
            self.sampling_offsets.bias.copy_(bias_flat)
            nn.init.constant_(self.sampling_offsets.weight, 0.0)
            nn.init.constant_(self.attn_weights.bias, 0.0)
            nn.init.constant_(self.attn_weights.weight, 0.0)
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, feat_maps, boxes, padding_mask=None, box_mask=None):
        """
        query: [B, Nq, C]
        feat_maps: list of n_levels feature maps, each [B, C, H_l, W_l]
        boxes: [B, Nq, 4] normalized coords
        padding_mask: [B, H_img, W_img] float (1 valid, 0 padding) OR None
        box_mask: [B, Nq] (optional) 1 valid, 0 padded
        returns: [B, Nq, C]
        """

        B, Nq, C = query.shape
        device = query.device
        assert len(feat_maps) == self.n_levels

        if box_mask is None:
            box_mask = torch.ones((B, Nq), dtype=torch.float32, device=device)

        # project query
        query_flat = query.view(B * Nq, C)  # [B*Nq, C]
        query_proj = self.query_proj(query_flat)  # [B*Nq, C]

        # predict offsets & attn
        sampling_offsets = self.sampling_offsets(query_proj)  # [B*Nq, n_h*n_p*2*n_l]
        #sampling_offsets = sampling_offsets.view(B, Nq, self.n_heads, self.n_points, 2, self.n_levels)  # [B,Nq,heads,points,2,levels]
        sampling_offsets = sampling_offsets.view(B, Nq, self.n_heads, self.n_levels, self.n_points, 2)

        attn = self.attn_weights(query_proj)  # [B*Nq, n_h*n_p*n_l]
        attn = attn.view(B, Nq, self.n_heads, self.n_levels*self.n_points)

        #attn = F.softmax(attn, dim=-1).view(B, Nq, self.n_heads, self.n_points, self.n_levels)
        attn = F.softmax(attn, dim=-1).view(B, Nq, self.n_heads, self.n_levels, self.n_points)

        #attn = attn.view(B, Nq, self.n_heads, self.n_points, self.n_levels)  # [B,Nq,heads,points,levels]
        # softmax over points dim
        #attn = F.softmax(attn, dim=3)  # [B,Nq,heads,points,levels]

        # box centers and sizes
        cx = (boxes[..., 0] + boxes[..., 2]) * 0.5  # [B,Nq]
        cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
        bw = (boxes[..., 2] - boxes[..., 0]).clamp(min=1e-6)
        bh = (boxes[..., 3] - boxes[..., 1]).clamp(min=1e-6)

        sampled_levels = []

        # prepare padding_mask if provided (to be sampled per grid)
        if padding_mask is not None:
            # padding_mask: [B, H_img, W_img] -> make channel dim
            pad_mask_in = padding_mask.unsqueeze(1).to(dtype=feat_maps[0].dtype)  # [B,1,H_img,W_img]
        else:
            pad_mask_in = None

        for lvl_idx, fm in enumerate(feat_maps):
            # fm: [B, C, H, W]
            Bf, Cf, H, W = fm.shape
            assert Bf == B and Cf == C

            # project value then we'll split into heads
            value = self.value_projs[lvl_idx](fm)  # [B, C, H, W]

            # get offsets for this level: [B,Nq,heads,points,2]
            offsets_lvl = sampling_offsets[..., :, :, :, lvl_idx]  # [B,Nq,heads,points,2]

            # scale offsets relative to box size -> normalized coords [0,1]
            bw_exp = bw.view(B, Nq, 1, 1)  # [B,Nq,1,1,1]
            bh_exp = bh.view(B, Nq, 1, 1)
            grid_x = cx.view(B, Nq, 1, 1) + offsets_lvl[..., 0] * bw_exp  # [B,Nq,heads,points,1]
            grid_y = cy.view(B, Nq, 1, 1) + offsets_lvl[..., 1] * bh_exp

            # clamp to [0,1]
            grid_x = grid_x.clamp(0.0, 1.0)
            grid_y = grid_y.clamp(0.0, 1.0)

            # convert to grid_sample coords [-1,1]
            grid_x_sample = grid_x * 2.0 - 1.0
            grid_y_sample = grid_y * 2.0 - 1.0

            # flatten points: we choose ordering so that later reshape matches
            # current ordering: [B, Nq, heads, points] -> we want flattened per image: n_pts = Nq * heads * points
            n_pts_per_image = Nq * self.n_heads * self.n_points
            # create gx,gy with ordering (B, n_pts)
            # first permute to [B, heads, points, Nq], then flatten to [B, n_pts]
            gx = grid_x_sample.permute(0, 2, 3, 1).contiguous().view(B, n_pts_per_image)
            gy = grid_y_sample.permute(0, 2, 3, 1).contiguous().view(B, n_pts_per_image)

            # grid for grid_sample: [B, n_pts, 1, 2] with (x,y) last
            grid = torch.stack((gx, gy), dim=-1).view(B, n_pts_per_image, 1, 2)

            # sample value: [B,C,H,W], grid [B,n_pts,1,2] -> [B,C,n_pts,1]
            print(value.shape, grid.shape)
            sampled = F.grid_sample(value, grid, mode='bilinear', align_corners=False, padding_mode='zeros')  # [B,C,n_pts,1]
            sampled = sampled.view(B, C, n_pts_per_image)  # [B,C,n_pts]
            # to [B, n_pts, C]
            sampled = sampled.permute(0, 2, 1)  # [B, n_pts, C]
            # reshape to [B, heads, points, Nq, C] because we flattened as (heads,points,Nq)
            sampled = sampled.view(B, self.n_heads, self.n_points, Nq, C)
            # permute to [B, Nq, heads, points, C]
            sampled = sampled.permute(0, 3, 1, 2, 4).contiguous()  # [B,Nq,heads,points,C]

            # split last dim C into (heads, head_dim) consistent per-head features
            # currently sampled has full embed_dim C per point per head; we need per-head head_dim
            # so we reshape sampled -> [B,Nq,heads,points,heads,head_dim]? that's redundant
            # Instead: because value projection produced embed_dim channels, we can reshape the C dim into (heads, head_dim)
            sampled = sampled.view(B, Nq, self.n_heads, self.n_points, self.n_heads, self.head_dim)
            # collapse the duplicate head axis by selecting diagonal: keep per-head slice at same index
            # To extract per-head features correctly: we should have produced value with shape [B, heads*head_dim, H, W]
            # After grid_sample our sampled tensor layout is [B, Nq, heads, points, C], where C = heads*head_dim.
            # We want to reshape last dim into [heads, head_dim] and then keep that per corresponding head index.
            # Use reshape and then take diagonal along the two head dims.
            sampled = sampled.permute(0,1,2,4,3,5).contiguous()  # [B,Nq,heads,heads,points,head_dim]
            # now take diag along the two head dims: i.e., for head h, pick sampled[..., h, h, :, :]
            # construct indices to gather diagonal
            # We'll produce an index tensor to pick diagonal efficiently:
            idx = torch.arange(self.n_heads, device=device)
            # sampled[..., head_idx, head_idx, points, head_dim]
            sampled_per_head = sampled[:, :, idx, idx, :, :].contiguous()  # [B,Nq,heads,points,head_dim]

            # now sampled_per_head is [B,Nq,heads,points,head_dim] as desired

            # sample padding mask (if provided) at same grid to get validity per sample point
            if pad_mask_in is not None:
                # pad_mask_in: [B,1,H_img,W_img] ; sample similarly
                sampled_mask = F.grid_sample(pad_mask_in, grid, mode='bilinear', align_corners=False, padding_mode='zeros')  # [B,1,n_pts,1]
                sampled_mask = sampled_mask.view(B, n_pts_per_image)  # [B, n_pts]
                sampled_mask = sampled_mask.view(B, self.n_heads, self.n_points, Nq)  # [B,heads,points,Nq]
                sampled_mask = sampled_mask.permute(0, 3, 1, 2).contiguous()  # [B,Nq,heads,points]
                sampled_mask = sampled_mask.unsqueeze(-1)  # [B,Nq,heads,points,1]
            else:
                sampled_mask = torch.ones((B, Nq, self.n_heads, self.n_points, 1), device=device)

            # attn weights for this level: [B,Nq,heads,points]
            attn_lvl = attn[..., lvl_idx]  # [B,Nq,heads,points]
            attn_lvl = attn_lvl.unsqueeze(-1)  # [B,Nq,heads,points,1]

            # apply padding mask: zero-out weights at padded sample points, then renormalize over points
            attn_masked = attn_lvl * sampled_mask  # [B,Nq,heads,points,1]
            sum_pts = attn_masked.sum(dim=3, keepdim=True)  # [B,Nq,heads,1,1]
            # avoid divide by zero: if sum_pts==0, keep attn_masked zeros (we add eps)
            eps = 1e-6
            attn_normalized = attn_masked / (sum_pts + eps)

            # weighted sum over points -> [B,Nq,heads,head_dim]
            weighted = (sampled_per_head * attn_normalized).sum(dim=3)  # sum over points

            # concat heads: reshape [B,Nq,heads,head_dim] -> [B,Nq, heads*head_dim==embed_dim]
            merged = weighted.view(B, Nq, self.n_heads * self.head_dim)  # [B,Nq,embed_dim]

            sampled_levels.append(merged)  # [B,Nq,embed_dim]

        # aggregate across levels
        feat_out = sum(sampled_levels) / float(self.n_levels)  # [B,Nq,embed_dim]

        # final linear
        out = self.output_proj(feat_out.view(B * Nq, C)).view(B, Nq, C)  # [B,Nq,C]
        return out


from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# -----------------------------
# Corrected MS-Deformable Cross Attention:
# - boxes are in original image pixel coords (x_min,y_min,x_max,y_max)
# - feature maps are downsampled versions of the original image: level l has scale s_l (e.g. 4,8,16,32)
# - we map original-image coords -> feature-map coords correctly for each level
# - attn softmax across (points * n_levels) per head
# - multi-head implemented by splitting channels into heads and using B*heads for grid_sample
# -----------------------------
class MSDeformableCrossAttentionCorrect(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3, level_strides=(8,16,32), base_scale=0.5):
        """
        level_strides: tuple of ints, downsample factor of each feature map relative to original image.
                       e.g. (4,8,16,32)
        boxes assumed in pixel coords relative to the image that was fed into backbone (H_img,W_img).
        """
        super().__init__()
        assert embed_dim % n_heads == 0
        assert len(level_strides) == n_levels
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.head_dim = embed_dim // n_heads
        self.level_strides = tuple(level_strides)
        self.base_scale = base_scale

        # linear layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, n_heads * n_points * 2 * n_levels)
        # produce attn for each head over (points * levels)
        self.attn_weights = nn.Linear(embed_dim, n_heads * n_points * n_levels)
        # per-level value convs
        self.value_projs = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(n_levels)])
        # output projection after concat heads
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_bias_2d()

    def _init_bias_2d(self):
        """Initialize sampling_offsets.bias with 2D grid per head per level (small phase shift)."""
        n_h, n_p, n_l = self.n_heads, self.n_points, self.n_levels
        grid_rows = int(math.floor(math.sqrt(n_p)))
        grid_cols = int(math.ceil(n_p / grid_rows))
        xs = torch.linspace(-0.5, 0.5, steps=grid_cols)
        ys = torch.linspace(-0.5, 0.5, steps=grid_rows)
        base_pts = []
        for yi in ys:
            for xi in xs:
                base_pts.append((xi.item(), yi.item()))
        base_pts = base_pts[:n_p]
        bias = torch.zeros((n_h, n_p, 2, n_l), dtype=torch.float32)
        for lvl in range(n_l):
            scale = self.base_scale * (1.0 + lvl / max(1, n_l - 1))
            for h in range(n_h):
                phase = (h / n_h) * (1.0 / max(1, n_p))
                for p_idx, (bx, by) in enumerate(base_pts):
                    shift_x = phase * ((p_idx % n_p) / n_p)
                    shift_y = -phase * ((p_idx % n_p) / n_p)
                    bias[h, p_idx, 0, lvl] = bx * scale + shift_x
                    bias[h, p_idx, 1, lvl] = by * scale + shift_y
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias.reshape(-1))
            nn.init.constant_(self.sampling_offsets.weight, 0.0)
            nn.init.constant_(self.attn_weights.weight, 0.0)
            nn.init.constant_(self.attn_weights.bias, 0.0)
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, feat_maps, boxes, padding_mask=None, box_mask=None):
        """
        query: [B, Nq, C]  (C == embed_dim)
        feat_maps: list(len=n_levels) of tensors [B, C, H_l, W_l]
        boxes: [B, Nq, 4] 必须是归一化的 (x_min, y_min, x_max, y_max) 在 [0, 1]
        padding_mask: optional [B, H_img, W_img] with 1=valid,0=pad
        box_mask: optional [B, Nq] (1 valid, 0 padded)
        returns: [B, Nq, C]
        """
        B, Nq, C = query.shape
        device = query.device

        if box_mask is None:
            box_mask = torch.ones((B, Nq), dtype=torch.float32, device=device)

        q = self.query_proj(query.view(B * Nq, C))  # [B*Nq, C]
        sampling_offsets = self.sampling_offsets(q).view(B, Nq, self.n_heads, self.n_points, 2, self.n_levels)

        attn_raw = self.attn_weights(q).view(B, Nq, self.n_heads, self.n_points * self.n_levels)
        attn_flat = F.softmax(attn_raw, dim=-1)  # [B,Nq,heads,P_total]

        # 此时的 boxes 已经是 [0, 1] 之间的归一化坐标
        x_min = boxes[..., 0]; y_min = boxes[..., 1]; x_max = boxes[..., 2]; y_max = boxes[..., 3]
        cx = (x_min + x_max) * 0.5  # [B,Nq]
        cy = (y_min + y_max) * 0.5
        bw = (x_max - x_min).clamp(min=1e-6)
        bh = (y_max - y_min).clamp(min=1e-6)

        if padding_mask is not None:
            pad_mask_in = padding_mask.unsqueeze(1).to(dtype=feat_maps[0].dtype)  # [B,1,H_img,W_img]
        else:
            pad_mask_in = None

        sampled_per_level = []
        mask_per_level = []

        for lvl_idx, fm in enumerate(feat_maps):
            Bf, Cf, H_l, W_l = fm.shape
            assert Bf == B and Cf == C

            value = self.value_projs[lvl_idx](fm)
            value = value.view(B, self.n_heads, self.head_dim, H_l, W_l)
            value_for_grid = value.view(B * self.n_heads, self.head_dim, H_l, W_l)

            # offs 也是相对于目标框宽高的比例
            offs = sampling_offsets[..., :, :, :, lvl_idx]

            # 直接在归一化坐标系 [0, 1] 下计算绝对偏移（无需相乘 image_size）
            off_x_norm = offs[..., 0] * bw.view(B, Nq, 1, 1)
            off_y_norm = offs[..., 1] * bh.view(B, Nq, 1, 1)

            # 在 [0, 1] 范围内限制采样点不越界
            x_sample_norm = (cx.view(B, Nq, 1, 1) + off_x_norm).clamp(0.0, 1.0)
            y_sample_norm = (cy.view(B, Nq, 1, 1) + off_y_norm).clamp(0.0, 1.0)

            # 转换为 grid_sample 专用的统一基准坐标系 [-1, 1]
            x_grid = x_sample_norm * 2.0 - 1.0
            y_grid = y_sample_norm * 2.0 - 1.0

            grid = torch.stack((x_grid, y_grid), dim=-1)  # [B,Nq,heads,points,2]
            grid = grid.permute(0, 2, 1, 3, 4).contiguous()
            n_pts = Nq * self.n_points
            grid = grid.view(B * self.n_heads, n_pts, 2).unsqueeze(2)

            sampled = F.grid_sample(value_for_grid, grid, mode='bilinear', align_corners=False)
            sampled = sampled.view(B, self.n_heads, self.head_dim, n_pts).permute(0, 3, 1, 2).contiguous()
            sampled = sampled.view(B, Nq, self.n_points, self.n_heads, self.head_dim).permute(0,1,3,2,4).contiguous()
            sampled_per_level.append(sampled)

            if pad_mask_in is not None:
                pad_for_grid = pad_mask_in.repeat(1, self.n_heads, 1, 1).view(B * self.n_heads, 1, pad_mask_in.shape[2], pad_mask_in.shape[3])
                # mask 的采样由于坐标已经是全图归一化的 [-1,1]，可以直接共用刚才的 grid
                sampled_mask = F.grid_sample(pad_for_grid, grid, mode='bilinear', align_corners=True)
                sampled_mask = sampled_mask.view(B, self.n_heads, n_pts).permute(0,2,1).contiguous()
                sampled_mask = sampled_mask.view(B, Nq, self.n_points, self.n_heads).permute(0,1,3,2).contiguous()
            else:
                sampled_mask = torch.ones((B, Nq, self.n_heads, self.n_points), dtype=sampled.dtype, device=device)

            mask_per_level.append(sampled_mask)

        mask_stack = torch.stack(mask_per_level, dim=-1)
        mask_flat = mask_stack.view(B, Nq, self.n_heads, self.n_points * self.n_levels)

        attn_masked = attn_flat * mask_flat
        sum_masked = attn_masked.sum(dim=-1, keepdim=True)
        eps = 1e-6
        attn_masked_norm = attn_masked / (sum_masked + eps)

        cond = (sum_masked <= eps)
        tiny_tensor = torch.full_like(attn_masked_norm, -1e20)
        attn_normalized_flat = torch.where(cond, tiny_tensor, attn_masked_norm)

        attn_normalized = attn_normalized_flat.view(B, Nq, self.n_heads, self.n_points, self.n_levels)

        weighted_levels = []
        for lvl_idx in range(self.n_levels):
            w_lvl = attn_normalized[..., :, lvl_idx].unsqueeze(-1)
            sampled_lvl = sampled_per_level[lvl_idx]
            weighted_lvl = (sampled_lvl * w_lvl).sum(dim=3)
            weighted_levels.append(weighted_lvl)

        weighted = sum(weighted_levels) / float(self.n_levels)
        merged = weighted.view(B, Nq, self.n_heads * self.head_dim)

        out = self.output_proj(merged.view(B * Nq, C)).view(B, Nq, C)
        return out



# md layer
class MD_layer(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3, mf_dim=[256, 256, 256], level_strides=[8, 16, 32]):
        super().__init__()
        
        self.rex_self_attn = Transformer(embed_dim, 4, heads=n_heads, mlp_dim=128, dim_head=64)
        self.sdropout = nn.Dropout(0.0)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.rex_cross_attn = MSDeformableCrossAttentionCorrect(embed_dim=embed_dim, n_heads=n_heads, n_points=n_points, n_levels=n_levels, base_scale=0.8, level_strides=level_strides)
        self.cdropout = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim*2)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim*2, embed_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(embed_dim)
    
    # 彻底移除 image_size 参数
    def forward(self, box_features, ibfeatures, tboxes_norm, padding_mask=None):
        # 传递归一化坐标 tboxes_norm
        box_features = box_features + self.cdropout(self.rex_cross_attn(box_features, ibfeatures, tboxes_norm, padding_mask=padding_mask))
        box_features = self.norm2(box_features)

        box_features = box_features + self.sdropout(self.rex_self_attn(box_features))        
        box_features = self.norm1(box_features)

        ffn_in = box_features
        ffn_out = self.linear2(self.dropout1(self.activation(self.linear1(box_features))))
        box_features = ffn_in + self.dropout2(ffn_out)
        box_features = self.norm3(box_features)

        return box_features


class MMTREX(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3, mf_dim=[256, 256, 256], level_strides=[8, 16, 32], n_depth=4, multi_head=False):
        super().__init__()
        self.rex_pos_embed = BoxPositionalEncoding(embed_dim)
        
        self.md_layer = nn.ModuleList([MD_layer(embed_dim=embed_dim, n_heads=n_heads, n_levels=n_levels, n_points=n_points, mf_dim=mf_dim, level_strides=level_strides) for _ in range(n_depth)])

        assert n_levels == len(mf_dim)

        self.rex_mf_proj = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(mf_dim[im], embed_dim, 1, 1),
            )
            for im in range(n_levels)
        ])
    
        self.multi_head = multi_head

    def forward(self, tboxes, ibfeatures, padding_masks=None):
        flag = 0
        if len(tboxes.size()) == 2:
            tboxes = tboxes.unsqueeze(0) # [1, n, 4]
            flag = 1

        # 核心修改：直接从 [cx, cy, w, h] 转为 [x_min, y_min, x_max, y_max]，并且它仍然是 [0,1] 的归一化形式！
        tboxes_norm = box_cxcywh_to_xyxy(tboxes)
        
        # 不要乘以任何 image_size 放缩！彻底保留在 [0, 1]！
        box_features = self.rex_pos_embed(tboxes_norm) # 位置编码完美适配归一化输入
        ibfeatures = [self.rex_mf_proj[im](ibfeatures[im]) for im in range(len(ibfeatures))]
        
        abox_features = []
        for layer in self.md_layer:
            # 查特征也完美利用这组归一化坐标，无需传入 image_size
            box_features = layer(box_features, ibfeatures, tboxes_norm, padding_mask=padding_masks)
            
            if self.multi_head:
                abox_features.append(box_features)

        if flag == 1:
            box_features = box_features.squeeze(0)
        
        if self.multi_head and flag==1:
            abox_features = [e.squeeze(0) for e in abox_features]

        return box_features


class MTREX(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=4, n_levels=3, mf_dim=[256, 256, 256], level_strides=[8, 16, 32]):
        #
        # self.attn
        super().__init__()
        self.rex_self_attn = Transformer(embed_dim, 12, heads=n_heads, mlp_dim=128, dim_head=64)
        self.sdropout = nn.Dropout(0.0)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # cross-attn
        self.rex_pos_embed = BoxPositionalEncoding(embed_dim)
        self.rex_cross_attn = MSDeformableCrossAttentionCorrect(embed_dim=embed_dim, n_heads=n_heads, n_points=n_points, n_levels=n_levels, base_scale=0.8, level_strides=level_strides)
        self.cdropout = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(embed_dim)

        assert n_levels == len(mf_dim)
        self.rex_mf_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mf_dim[im], embed_dim, 1, 1),
            )
            for im in range(n_levels)

        ])
        
        # ffn
        self.linear1 = nn.Linear(embed_dim, embed_dim*2)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim*2, embed_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(embed_dim)


    def forward(self, tboxes, ibfeatures, padding_masks=None):
        flag = 0
        if len(tboxes.size()) == 2:
            tboxes = tboxes.unsqueeze(0) # [1, n, 4]
            flag = 1

        #从cxcywh转换到xyxy，且不归一化
        tboxes = box_cxcywh_to_xyxy(tboxes)
        image_h, image_w = 640, 640
        scale_tensor = torch.tensor([image_w, image_h, image_w, image_h], device=tboxes.device, dtype=tboxes.dtype)
        tboxes = tboxes * scale_tensor
        
        # tboxes: 归一化， [n, 4]
        box_features = self.rex_pos_embed(tboxes/640.) # [b, n, c]
        ibfeatures = [self.rex_mf_proj[im](ibfeatures[im]) for im in range(len(ibfeatures))]


        # cross-attn
        box_features = box_features + self.cdropout(self.rex_cross_attn(box_features, ibfeatures, tboxes, image_size=[640, 640], padding_mask=padding_masks))
        box_features = self.norm2(box_features)
        
        # 先做self.attn
        box_features = box_features + self.sdropout(self.rex_self_attn(box_features))
        box_features = self.norm1(box_features)
        
        # ffn
        box_features = self.linear2(self.dropout1(self.activation(self.linear1(box_features))))
        box_features = box_features + self.dropout2(box_features)
        box_features = self.norm3(box_features)
        
        if flag == 1:
            box_features = box_features.squeeze(0)

        return box_features
        

