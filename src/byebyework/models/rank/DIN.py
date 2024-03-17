import torch
import torch.nn as nn

from byebyework.models import active_function as m

class DIN(nn.Module):

    def __init__(self,user_num,item_num,cate_num,hidden_size=64):
        """
        DIN input parameters
        :param user_num: int numbers of users
        :param item_num: int numbers of items
        :param cate_num: int numbers of categories
        :param hidden_size: embedding_size
        """
        super(DIN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.u_emb = nn.Embedding(user_num, hidden_size)
        self.i_emb = nn.Embedding(item_num, hidden_size)
        self.c_emb = nn.Embedding(cate_num, hidden_size)
        self.linear =  nn.Sequential(
            nn.Linear(hidden_size * 4, 80),
            m.Dice(80),
            nn.Linear(80, 40),
            m.Dice(40),
            nn.Linear(40, 2)
        )
        self.au = m.ActivationUnit(hidden_size)
        
    def forward(self,user,hist,item,cate):
        """
        :param user: user id
        :param hist: list of history behaviors of user
        :param item: item id
        :param cate: category id of item
        """
        user = self.u_emb(user).squeeze()
        item = self.i_emb(item).squeeze()
        cate = self.c_emb(cate).squeeze()
        h = []
        weights = []
        for i in range(len(hist)):
            hist_i = self.i_emb(hist[i])
            h.append(hist_i.squeeze().detach().numpy())
            weight = self.au(hist_i,item)
            weights.append(weight)
            
        cur = torch.zeros_like(h[0])
        for i in range(len(h)):
            cur += torch.tensor(weights[i] * h[i], dtype=torch.float32)
            
        res = torch.cat([user,item,cate,cur],-1)
        res = self.linear(res)
        return res