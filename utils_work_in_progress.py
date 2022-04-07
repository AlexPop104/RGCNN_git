def get_knn_adj_matrix_pytorch(x,batch,k):

    edge_index = knn_graph(x, k=k, batch=batch, loop=True)

    pcd1=x[edge_index[0]]
    pcd2=x[edge_index[1]]

    point_cloud_inner = torch.sum(torch.mul(pcd1, pcd2),dim=1, keepdim=True)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square_1 = torch.sum(torch.mul(pcd1, pcd1), dim=1, keepdim=True)
    point_cloud_square_2 = torch.sum(torch.mul(pcd2, pcd2), dim=1, keepdim=True)
    distances = point_cloud_square_1 + point_cloud_inner + point_cloud_square_2
    distances=distances.squeeze(1)

    distances=torch.exp(-distances)

    adj_matrix_2=tg.utils.to_dense_adj(edge_index=edge_index,edge_attr= distances,batch=batch)

    return adj_matrix_2


def get_RotationInvariantFeatures(point_cloud,num_points):

    nr_coordinates=point_cloud.shape[2]
    batch_size=point_cloud.shape[0]
    
    centroid=torch.sum(point_cloud,1)
    centroid=centroid/num_points

    centroid_pos=torch.tile(centroid[:,0:3],(1,num_points))
    centroid_pos=torch.reshape(centroid_pos,(batch_size,num_points,3))

    centroid_norm=torch.tile(centroid[:,3:6],(1,num_points))
    centroid_norm=torch.reshape(centroid_pos,(batch_size,num_points,3))


    Pos_dif=torch.subtract(point_cloud[:,:,0:3],centroid_pos)
    

    Distances=torch.linalg.norm(Pos_dif,dim=2)
    Distances=torch.unsqueeze(Distances,2)

    cosine_sim=torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    output_1 = cosine_sim(centroid_norm, point_cloud[:,:,3:6])
    output_1=torch.unsqueeze(output_1,2)
    output_2 = cosine_sim(centroid_norm, Pos_dif)
    output_2=torch.unsqueeze(output_2,2)
    output_3=cosine_sim(point_cloud[:,:,3:6],Pos_dif)
    output_3=torch.unsqueeze(output_3,2)
    
    PPF_features=torch.cat((Distances,output_1,output_2,output_3),dim=2)


    return PPF_features

def get_fps_matrix_topk(point_cloud,batch_size,nr_points,nr_points_fps):
    nr_points_batch=nr_points

    Batch_indexes=torch.arange(0,batch_size,device='cuda')
    Batch_indexes=torch.reshape(Batch_indexes,[batch_size,1])
    Batch_indexes=torch.tile(Batch_indexes,(1,nr_points_batch))
    Batch_indexes=torch.reshape(Batch_indexes,[batch_size*nr_points_batch])

    index = fps(point_cloud, Batch_indexes, ratio=float(nr_points_fps/nr_points) , random_start=True)

    fps_point_cloud=point_cloud[index]
    fps_batch=Batch_indexes[index]

    fps_point_cloud_2=torch.tile(fps_point_cloud,(1,nr_points))

    point_cloud_2=torch.reshape(point_cloud,(batch_size,nr_points,point_cloud.shape[1]))

    point_cloud_3=torch.reshape(point_cloud_2,(batch_size,nr_points*point_cloud.shape[1]))

    point_cloud_4=torch.tile(point_cloud_3,(1,nr_points_batch))

    ##Work in progress
    
    return fps_batch

def test_pcd_pred(model, loader,num_points,device):
    with torch.no_grad():
        label_to_names = {0: 'airplane',
                                1: 'bathtub',
                                2: 'bed',
                                3: 'bench',
                                4: 'bookshelf',
                                5: 'bottle',
                                6: 'bowl',
                                7: 'car',
                                8: 'chair',
                                9: 'cone',
                                10: 'cup',
                                11: 'curtain',
                                12: 'desk',
                                13: 'door',
                                14: 'dresser',
                                15: 'flower_pot',
                                16: 'glass_box',
                                17: 'guitar',
                                18: 'keyboard',
                                19: 'lamp',
                                20: 'laptop',
                                21: 'mantel',
                                22: 'monitor',
                                23: 'night_stand',
                                24: 'person',
                                25: 'piano',
                                26: 'plant',
                                27: 'radio',
                                28: 'range_hood',
                                29: 'sink',
                                30: 'sofa',
                                31: 'stairs',
                                32: 'stool',
                                33: 'table',
                                34: 'tent',
                                35: 'toilet',
                                36: 'tv_stand',
                                37: 'vase',
                                38: 'wardrobe',
                                39: 'xbox'}
        for data in loader:
                # x=data.pos
                # x=x.reshape(data.batch.unique().shape[0], num_points, 3)
                # x2=get_centroid(point_cloud=x,num_points=num_points)

                x = torch.cat([data.pos, data.normal], dim=1)   
                x = x.reshape(data.batch.unique().shape[0], num_points, 6)

                L = pairwise_distance(x)
                #L = get_laplacian(L)

                # x=torch.cat([x,x2],dim=2)
        
                #logits, regularizers  = model(x=x.to(device),x2=x2.to(device))
                logits, regularizers  = model(x=x.to(device))


                viz_points=data.pos
                viz_points=viz_points.reshape(data.batch.unique().shape[0], num_points, 3)

                viz_points=viz_points.to('cuda')

                pred = logits.argmax(dim=-1)

                ground_truth=data.y.to(device)
                
                for it_pcd in range(data.batch.unique().shape[0]):
                    if(ground_truth[it_pcd]==pred[it_pcd]):

                        print("Actual label:")
                        print(label_to_names[ground_truth[it_pcd].item()])
                        print("Predicted label:")
                        print(label_to_names[pred[it_pcd].item()])

                        viz_points_2=viz_points[it_pcd,:,:]
                        distances=L[it_pcd,:,:]

                        distances=distances.to('cuda')

                        threshold=0.7

                        #view_graph(viz_points_2,distances,threshold,it_pcd)

                        fig = plt.figure(label_to_names[pred[it_pcd].item()])
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_axis_off()
                        for i in range(viz_points.shape[1]):
                            ax.scatter(viz_points[it_pcd,i,0].item(),viz_points[it_pcd,i, 1].item(), viz_points[it_pcd,i,2].item(),color='r')

                        # #distances=distances-distances.min()
                        
                        # distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype))

                        # for i in range(viz_points.shape[1]):
                        #     for j in range(viz_points.shape[1]):
                        #         if (distances[i,j].item()>0.4):
                        #             ax.plot([viz_points[it_pcd,i,0].item(),viz_points[it_pcd,j,0].item()],[viz_points[it_pcd,i, 1].item(),viz_points[it_pcd,j, 1].item()], [viz_points[it_pcd,i,2].item(),viz_points[it_pcd,j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))
                            
                        plt.show()

                #plt.show()

def view_graph(viz_points,distances,threshold,nr):

    distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype,device='cuda'))

    fig = plt.figure(nr)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    for i in range(viz_points.shape[0]):
        ax.scatter(viz_points[i,0].item(),viz_points[i, 1].item(), viz_points[i,2].item(),color='r')

    for i in range(viz_points.shape[0]):
        for j in range(viz_points.shape[0]):
            if (distances[i,j].item()>threshold):
                ax.plot([viz_points[i,0].item(),viz_points[j,0].item()],[viz_points[i, 1].item(),viz_points[j, 1].item()], [viz_points[i,2].item(),viz_points[j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))

def view_graph_with_original_pcd(viz_points,original_points,distances,threshold,nr):

    distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype,device='cuda'))

    fig = plt.figure(nr)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    for i in range(viz_points.shape[0]):
        ax.scatter(viz_points[i,0].item(),viz_points[i, 1].item(), viz_points[i,2].item(),color='r')

    for i in range(original_points.shape[0]):
        ax.scatter(original_points[i,0].item(),original_points[i, 1].item(), original_points[i,2].item(),color='g')

    for i in range(viz_points.shape[0]):
        for j in range(viz_points.shape[0]):
            if (distances[i,j].item()>threshold):
                ax.plot([viz_points[i,0].item(),viz_points[j,0].item()],[viz_points[i, 1].item(),viz_points[j, 1].item()], [viz_points[i,2].item(),viz_points[j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))

def view_graph_Reeb(viz_points,Reeb_points,distances,threshold,nr):

    distances=torch.where(distances<1,distances,torch.tensor(1., dtype=distances.dtype,device='cuda'))

    fig = plt.figure(nr)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    

    for t in range(Reeb_points.shape[0]):
        ax.scatter(Reeb_points[t,0].item(),Reeb_points[t, 1].item(), Reeb_points[t,2].item(),color='g')

    for k in range(viz_points.shape[0]):
        ax.scatter(viz_points[k,0].item(),viz_points[k, 1].item(), viz_points[k,2].item(),color='r')    

    for i in range(Reeb_points.shape[0]):
        for j in range(Reeb_points.shape[0]):
            if (distances[i,j].item()>threshold):
                ax.plot([Reeb_points[i,0].item(),Reeb_points[j,0].item()],[Reeb_points[i, 1].item(),Reeb_points[j, 1].item()], [Reeb_points[i,2].item(),Reeb_points[j,2].item()],alpha=1,color=(0,0,distances[i,j].item()))

def test_pcd_with_index(model,loader,num_points,device):
    with torch.no_grad():
        label_to_names = {0: 'airplane',
                                1: 'bathtub',
                                2: 'bed',
                                3: 'bench',
                                4: 'bookshelf',
                                5: 'bottle',
                                6: 'bowl',
                                7: 'car',
                                8: 'chair',
                                9: 'cone',
                                10: 'cup',
                                11: 'curtain',
                                12: 'desk',
                                13: 'door',
                                14: 'dresser',
                                15: 'flower_pot',
                                16: 'glass_box',
                                17: 'guitar',
                                18: 'keyboard',
                                19: 'lamp',
                                20: 'laptop',
                                21: 'mantel',
                                22: 'monitor',
                                23: 'night_stand',
                                24: 'person',
                                25: 'piano',
                                26: 'plant',
                                27: 'radio',
                                28: 'range_hood',
                                29: 'sink',
                                30: 'sofa',
                                31: 'stairs',
                                32: 'stool',
                                33: 'table',
                                34: 'tent',
                                35: 'toilet',
                                36: 'tv_stand',
                                37: 'vase',
                                38: 'wardrobe',
                                39: 'xbox'}
        for i,(pos, y, normal, idx) in enumerate(loader):
               
                viz_points=pos[1]
                viz_points=viz_points.reshape(pos[1].shape[0], num_points, 3)

                
                for it_pcd in range(pos[1].shape[0]):
                     
                        print("PCD nr:")
                        print(idx[it_pcd])
                        print("PCD label")
                        print(label_to_names[y[1][it_pcd].item()])
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_axis_off()
                        ax.scatter(viz_points[it_pcd,:,0], viz_points[it_pcd,:, 1], viz_points[it_pcd,:,2], s=1, color='r')   
                        plt.show()

def view_pcd(model,loader,num_points,device,program_name):
    with torch.no_grad():
        for i,data in enumerate(loader):
               
                viz_points=data.pos
                viz_points=viz_points.reshape(data.batch.unique().shape[0], num_points, 3)

                viz_normals=data.normal
                viz_normals=viz_normals.reshape(data.batch.unique().shape[0], num_points, 3)

                
                
                for it_pcd in range(data.pos.shape[0]):

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(viz_points[it_pcd,:,:])

                        
                        #o3d.geometry.PointCloud.estimate_normals(pcd)
                        pcd.colors=o3d.utility.Vector3dVector(viz_normals[it_pcd,:,:])
                        
                        pcd.normals=o3d.utility.Vector3dVector(viz_normals[it_pcd,:,:])

                        # pcd.paint_uniform_color([0, 0.651, 0.929])
                        o3d.visualization.draw_geometries([pcd],window_name=program_name, width=500, height=500,point_show_normal=True)
                        #o3d.visualization.draw_geometries([pcd],window_name=program_name, width=500, height=500)
                         

                        # fig = plt.figure(program_name)
                        # ax = fig.add_subplot(111, projection='3d')
                        # ax.set_axis_off()
                        # ax.scatter(viz_points[it_pcd,:,0], viz_points[it_pcd,:, 1], viz_points[it_pcd,:,2], s=1, color='r')   
                        # plt.show()

def get_label_Modelnet(position):
    label_to_names = {0: 'airplane',
                                1: 'bathtub',
                                2: 'bed',
                                3: 'bench',
                                4: 'bookshelf',
                                5: 'bottle',
                                6: 'bowl',
                                7: 'car',
                                8: 'chair',
                                9: 'cone',
                                10: 'cup',
                                11: 'curtain',
                                12: 'desk',
                                13: 'door',
                                14: 'dresser',
                                15: 'flower_pot',
                                16: 'glass_box',
                                17: 'guitar',
                                18: 'keyboard',
                                19: 'lamp',
                                20: 'laptop',
                                21: 'mantel',
                                22: 'monitor',
                                23: 'night_stand',
                                24: 'person',
                                25: 'piano',
                                26: 'plant',
                                27: 'radio',
                                28: 'range_hood',
                                29: 'sink',
                                30: 'sofa',
                                31: 'stairs',
                                32: 'stool',
                                33: 'table',
                                34: 'tent',
                                35: 'toilet',
                                36: 'tv_stand',
                                37: 'vase',
                                38: 'wardrobe',
                                39: 'xbox'}
    return(label_to_names[position])

class DenseChebConv_theta_and_sum(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv_theta_and_sum, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels , out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(1, 1, bias=False, 
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
        self.lin.reset_parameters()
        
        # self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        #x = x.unsqueeze if x.dim() == 2 else x
        #L = L.unsqueeze if L.dim() == 2 else L

        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin



        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,[x.shape[0]*x.shape[1]*x.shape[2]])
        x0=x0.unsqueeze(0)

        x0=torch.permute(x0,(1,0))


        # x = x0.unsqueeze(0)
        out = self.lins[0](x0)

        x0=x0.unsqueeze(1)
        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,(x.shape[0],x.shape[1],x.shape[2]))

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)

            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,[x.shape[0]*x.shape[1]*x.shape[2]])

            x1=x1.unsqueeze(0)
            x1=torch.permute(x1,(1,0))

            out = out + self.lins[1](x1)

            x1=x1.unsqueeze(1)
            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,(x.shape[0],x.shape[1],x.shape[2]))
            # x = concat(x, x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0

            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,[x.shape[0]*x.shape[1]*x.shape[2]])

            x2=x2.unsqueeze(0)
            x2=torch.permute(x2,(1,0))

            out += self.lins[i](x2)

            x2=x2.unsqueeze(1)
            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,(x.shape[0],x.shape[1],x.shape[2]))
            # x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        out = out.reshape([N * M, Fin])

        out = self.lin(out)
        out = out.reshape([N, M, self.out_channels])

        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')

class DenseChebConv_small_linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv_small_linear, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels, out_channels, bias=bias)
        
        self.reset_parameters()


    def reset_parameters(self):        
        self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        x = x.unsqueeze if x.dim() == 2 else x
        L = L.unsqueeze if L.dim() == 2 else L

        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin
        #x = x0.unsqueeze(0)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            x = torch.add(x, x1)
            #x = concat(x, x1)

        for _ in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            x = torch.add(x, x2)
            #x = concat(x, x2)
            x0, x1 = x1, x2

        # x = x.permute([1,2,3,0])
        x = x.reshape([N * M, Fin ])

        x = self.lin(x)
        x = x.reshape([N, M, self.out_channels])
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')

class DenseChebConv_theta_nosum(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConv_theta_nosum, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lin = Linear(in_channels * K, out_channels, bias=bias)
        self.lins = t.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False, 
                weight_initializer='glorot') for _ in range(K)
        ])
        self.lins_theta = t.nn.ModuleList([
            Linear(1, 1, bias=False, 
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
        for lin in self.lins_theta:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, 0, 0.1)
        
        # self.lin.weight = torch.nn.init.trunc_normal_(self.lin.weight, 0, 0.1)
        # self.lin.reset_parameters()


    def forward(self, x, L, mask=None):
        #x = x.unsqueeze if x.dim() == 2 else x
        #L = L.unsqueeze if L.dim() == 2 else L

        #N, M, Fin = x.shape
        #N, M, Fin = int(N), int(M), int(Fin)

        x0 = x # N x M x Fin

        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,[x.shape[0]*x.shape[1]*x.shape[2]])
        x0=x0.unsqueeze(0)

        x0=torch.permute(x0,(1,0))


        # x = x0.unsqueeze(0)
        out_theta = self.lins_theta[0](x0)

        x0=x0.unsqueeze(1)
        x0=torch.reshape(x0,(x.shape[0],x.shape[1]*x.shape[2]))
        x0=torch.reshape(x0,(x.shape[0],x.shape[1],x.shape[2]))

        out_theta=out_theta.unsqueeze(1)
        out_theta=torch.reshape(out_theta,(x.shape[0],x.shape[1],x.shape[2]))

        # x = x0.unsqueeze(0)
        out = self.lins[0](out_theta)

        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            return t.cat([x, x_], dim=0)

        if self.K > 1:
            x1 = t.matmul(L, x0)

            x1 = t.matmul(L, x0)

            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,[x.shape[0]*x.shape[1]*x.shape[2]])

            x1=x1.unsqueeze(0)
            x1=torch.permute(x1,(1,0))

            out_theta = self.lins_theta[1](x1)

            x1=x1.unsqueeze(1)
            x1=torch.reshape(x1,(x.shape[0],x.shape[1]*x.shape[2]))
            x1=torch.reshape(x1,(x.shape[0],x.shape[1],x.shape[2]))

            out_theta=out_theta.unsqueeze(1)
            out_theta=torch.reshape(out_theta,(x.shape[0],x.shape[1],x.shape[2]))

            out = out + self.lins[1](out_theta)
            # x = concat(x, x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0

            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,[x.shape[0]*x.shape[1]*x.shape[2]])

            x2=x2.unsqueeze(0)
            x2=torch.permute(x2,(1,0))

            out_theta = self.lins_theta[i](x2)

            x2=x2.unsqueeze(1)
            x2=torch.reshape(x2,(x.shape[0],x.shape[1]*x.shape[2]))
            x2=torch.reshape(x2,(x.shape[0],x.shape[1],x.shape[2]))

            out_theta=out_theta.unsqueeze(1)
            out_theta=torch.reshape(out_theta,(x.shape[0],x.shape[1],x.shape[2]))

            out =out + self.lins[i](out_theta)
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

class GaussianNoiseTransform(BaseTransform):

    def __init__(self, mu: Optional[float] = 0, sigma: Optional[float] = 0.1, recompute_normals : bool = True):
        torch.manual_seed(0)
        self.mu = mu
        self.sigma = sigma
        self.recompute_normals = recompute_normals

    def __call__(self, data: Union[Data, HeteroData]):
        noise = np.random.normal(self.mu, self.sigma, data.pos.shape)
        data.pos += noise
        data.pos = data.pos.float()
        if self.recompute_normals:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(data.pos)
            pcd_o3d.estimate_normals(fast_normal_computation=False)
            pcd_o3d.normalize_normals()
            if hasattr(data, 'normal'):
                data.normal = np.asarray(pcd_o3d.normals)
                data.normal = torch.tensor(data.normal, dtype=torch.float32)
            else:
                data.normal = np.asarray(pcd_o3d.normals)
                data.normal = torch.tensor(data.normal, dtype=torch.float32)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
