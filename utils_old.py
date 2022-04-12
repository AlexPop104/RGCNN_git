class DenseChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        
        self.reset_parameters()


    def reset_parameters(self):        
        self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        x = x.unsqueeze if x.dim() == 2 else x
        L = L.unsqueeze if L.dim() == 2 else L

        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin
        x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            x = concat(x, x1)

        for _ in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2

        x = x.permute([1,2,3,0])
        x = x.reshape([N * M, Fin * self.K])

        x = self.lin(x)
        x = x.reshape([N, M, self.out_channels])
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')

class DenseChebConvV2_old(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConvV2_old, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False, 
                weight_initializer='glorot') for _ in range(K)
        ])
        
        if bias:
            self.bias = Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):        
        for lin in self.lins:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, 0, 0.1)
        
        
        # self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        #x = x.unsqueeze if x.dim() == 2 else x
        #L = L.unsqueeze if L.dim() == 2 else L

        #N, M, Fin = x.shape
        #N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin
        # x = x0.unsqueeze(0)
        out = self.lins[0](x0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            out = out + self.lins[1](x1)
            # x = concat(x, x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            out += self.lins[i](x2)
            # x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        # x = x.reshape([N * M, Fin * self.K])

        # x = self.lin(x)
        # x = x.reshape([N, M, self.out_channels])

        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')