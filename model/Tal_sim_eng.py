from .encoder.Transformer import make_model
import torch.nn as nn
import torch.nn.functional as F
import torch

class qiao_infer(nn.Module):
    def __init__(self,d_model,number_block_1,head_number,d_ff,seq_len,vocab_size,drop_out):
        super(qiao_infer, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)

        self.encoder, self.word_embedding, self.pos_emb = make_model(N=number_block_1,d_model=d_model,h=head_number,d_ff=d_ff, seq_len=seq_len, vocab_size=vocab_size)
        # Concat Attention
        ## for hp
        self.Wc1 = nn.Linear(d_model,d_model,bias=False)
        self.Wc2 = nn.Linear(d_model,d_model,bias=False)
        self.vc = nn.Linear(d_model,1,bias=False)
        ## for hc
        self.Wc1_ = nn.Linear(d_model,d_model,bias=False)
        self.Wc2_ = nn.Linear(d_model,d_model,bias=False)
        self.vc_ = nn.Linear(d_model,1,bias=False)

        # Bilinear Attention
        ## for hp
        self.Wb = nn.Linear(d_model,d_model,bias=False)
        ## for hc
        self.Wb_ = nn.Linear(d_model,d_model,bias=False)
        
        # Dot Attention
        ## for hp
        self.Wd = nn.Linear(d_model,d_model,bias=False)
        self.vd = nn.Linear(d_model,1,bias=False)
        ## for hc
        self.Wd_ = nn.Linear(d_model,d_model,bias=False)
        self.vd_ = nn.Linear(d_model,1,bias=False)
        
        # Minus Attention
        ## for hp
        self.Wm = nn.Linear(d_model,d_model,bias=False)
        self.Vm = nn.Linear(d_model,1,bias=False)
        ## for hc
        self.Wm_ = nn.Linear(d_model,d_model,bias=False)
        self.Vm_ = nn.Linear(d_model,1,bias=False)

        # after multiway attention, use as pooling
        #self.W_agg_p = nn.Linear(4*d_model,d_model)
        #self.W_agg_p_ = nn.Linear(4*d_model,d_model)

        # self attention structure for left sentence
        self.Wc1_p = nn.Linear(4*d_model, d_model, bias=False)
        self.vc_p = nn.Linear(d_model, 1, bias=False)

        # self attention structure for right sentence
        self.Wc1_p_ = nn.Linear(4*d_model,d_model,bias=False)
        self.vc_p_ = nn.Linear(d_model,1,bias=False)
        
        # prediction
        self.MLP_layers = nn.Sequential(
            nn.Linear(4*4*d_model,d_model),
            nn.ReLU(),
            nn.Dropout(p = drop_out),
            nn.Linear(d_model,1)
            )
    def forward(self,post,comm):
        batch_size = post.shape[0]
        p_embedding = self.word_embedding(post)
        p_embedding = p_embedding + self.pos_emb(p_embedding)
        c_embedding = self.word_embedding(comm)
        c_embedding = c_embedding + self.pos_emb(c_embedding)
       
        hp = self.encoder(p_embedding)
        hp=self.dropout(hp)
        
        hc = self.encoder(c_embedding)
        hc=self.dropout(hc)

        # multiply hp
        _s1 = self.Wc1(hp).unsqueeze(1)
        _s2 = self.Wc2(hc).unsqueeze(2)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze(3)
        ait = F.softmax(sjt, 2)
        ptc = ait.bmm(hp)

        _s1 = self.Wb(hp).transpose(2, 1)
        sjt = hc.bmm(_s1)
        ait = F.softmax(sjt, 2)
        ptb = ait.bmm(hp)

        _s1 = hp.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze(3)
        ait = F.softmax(sjt, 2)
        ptd = ait.bmm(hp)

        sjt = self.Vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze(3)
        ait = F.softmax(sjt, 2)
        ptm = ait.bmm(hp)
                
        # multiply hc
        _s1 = self.Wc1_(hc).unsqueeze(1)
        _s2 = self.Wc2_(hp).unsqueeze(2)
        sjt = self.vc_(torch.tanh(_s1 + _s2)).squeeze(3)
        ait = F.softmax(sjt, 2)
        ptc_ = ait.bmm(hc)
        
        _s1 = self.Wb_(hc).transpose(2, 1)
        sjt = hp.bmm(_s1)
        ait = F.softmax(sjt, 2)
        ptb_ = ait.bmm(hc)
        
        _s1 = hc.unsqueeze(1)
        sjt = self.vd_(torch.tanh(self.Wd_(_s1 * _s2))).squeeze(3)
        ait = F.softmax(sjt, 2)
        ptd_ = ait.bmm(hc)
        _s2 = hp.unsqueeze(2)
        
        sjt = self.Vm_(torch.tanh(self.Wm_(_s1 - _s2))).squeeze(3)
        ait = F.softmax(sjt, 2)
        ptm_ = ait.bmm(hc)

        aggregation_p = torch.cat([ptc,ptb,ptd,ptm],2)
        aggregation_p_ = torch.cat([ptc_,ptb_,ptd_,ptm_],2)

        # self_attention to make left matrix in vector
        sj = F.softmax(self.vc_p(self.Wc1_p(aggregation_p)).transpose(2,1),2)
        rc = sj.bmm(aggregation_p)
        rc = rc.squeeze(1)
        # self_attention to make right matrix in vector
        sj_ = F.softmax(self.vc_p_(self.Wc1_p_(aggregation_p_)).transpose(2,1),2)
        rc_ = sj_.bmm(aggregation_p_)
        rc_ = rc_.squeeze(1)
        
        pair = torch.cat([rc,rc_,rc*rc_, torch.abs(rc-rc_)],1)
        score = torch.sigmoid(self.MLP_layers(pair))
        return score.view(batch_size)







