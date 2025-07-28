import argparse
import os
import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler.sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import emptyCache
import os
from sampler.sampler_core import ParallelSampler, TemporalGraphBlock


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='LASTFM')
parser.add_argument('--config', type=str, help='path to config file', default='/root/swift/config/TGN-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--model_eval', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)
parser.add_argument('--use_cpu_sample', action='store_true', default=False)

parser.add_argument('--reuse_ratio', type=float, default=0.9, help='reuse_ratio')
parser.add_argument('--train_conf', type=str, default='disk', help='name of stored model')
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
parser.add_argument('--substream_size', type=int, default=60000)
parser.add_argument('--set_epoch', type=int, default=-1, help='distance threshold')
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

from config.train_conf import *
GlobalConfig.conf = args.train_conf + '.json'
config = GlobalConfig()
args.use_async_prefetch = config.use_async_prefetch
args.use_async_IO = config.use_async_IO




print(sample_param)
print(train_param)
args.cut_zombie = config.cut_zombie
if (hasattr(config, 'model')):
    args.config = f'/root/swift/config/{config.model}-{config.layer}.yml'

if (config.model_eval):
    args.model_eval = True

print(f"train config: {config.config_data}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    print(f"======================================================================")
    print(f"set random seed:{seed}")
    print(f"======================================================================")
set_seed(42)

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds


def eval(mode='val'):
    
    if (emb_buffer):
        emb_buffer.cur_mode = 'val'
    if (feat_buffer):
        feat_buffer.mode = 'val'

    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()

    if mode == 'val':
        left = df_conf['train_edge_end']
        eval_df_end = val_edge_end
    elif mode == 'test':
        left = df_conf['val_edge_end']
        eval_df_end = datas['src'].shape[0]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        left = 0
        eval_df_end = df_conf['train_edge_end']
    
    right = left
    with torch.no_grad():
        total_loss = 0
        while True:
            right += batch_size
            right = min(right, eval_df_end)
            if (left >= right):
                break

            src = datas['src'][left: right]
            dst = datas['dst'][left: right]
            times = datas['time'][left: right]
            eid = datas['eid'][left: right]
            root_nodes = np.concatenate([src, dst, neg_link_sampler.sample(src.shape[0] * neg_samples)]).astype(np.int32)
            ts = np.tile(times, neg_samples + 2).astype(np.float32)
            
            if (use_gpu_sample):
                root_nodes = torch.from_numpy(root_nodes).cuda()
                root_ts = torch.from_numpy(ts).cuda()
                ret = sampler_gpu.sample_layer(root_nodes, root_ts, cut_zombie=args.cut_zombie)
            else:
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = len(rows) * 2
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()

            if (use_gpu_sample):
                mfgs = sampler_gpu.gen_mfgs(ret)
                root_nodes = root_nodes.cpu().numpy()
            else:
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'])
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts)  
                    
            mfgs = prepare_input(mfgs, node_feats, edge_feats, feat_buffer = feat_buffer, combine_first=combine_first)
            
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = eid.cuda().to(torch.int32)
                mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
            
            left = right

        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

# set_seed(0)
if __name__ == '__main__':

    try:
        global node_feats, edge_feats
        node_feats, edge_feats = None,None

        import torch.multiprocessing as multiprocessing
        multiprocessing.set_start_method("spawn")
        from pre_fetch import *
        use_async_prefetch = args.use_async_prefetch

        parent_conn_IO = None
        if (args.use_async_IO):
            parent_conn_IO, child_conn_IO = multiprocessing.Pipe()
            prefetch_conn_IO, prefetch_child_conn_IO = multiprocessing.Pipe()

            p = multiprocessing.Process(target=prefetch_worker_IO, args=(child_conn_IO, prefetch_child_conn_IO))
            p.start()

            

        parent_conn = None
        prefetch_conn = None
        # if (use_async_prefetch):
        parent_conn, child_conn = multiprocessing.Pipe()
        prefetch_conn, prefetch_child_conn = multiprocessing.Pipe()

        p = multiprocessing.Process(target=prefetch_worker, args=(child_conn, prefetch_child_conn))
        p.start()

        parent_conn.send(('init_feats', (args.data, args.substream_size )))
        print(f"Sent: {'init_feats'}")
        result = parent_conn.recv()
        print(f"Received: {result}")
        node_feats,edge_feats = 1,1

        if (args.use_async_IO):
            parent_conn.send(('init_IO_load', (parent_conn_IO,)))
            result = parent_conn.recv()

        
        
        g, datas, df_conf = load_graph_bin(args.data)

        train_edge_end = df_conf['train_edge_end']
        val_edge_end = df_conf['val_edge_end']

        if args.use_inductive:
            inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
            df = df.iloc[inductive_inds]
            

        if (args.data == 'LASTFM'):
            gnn_dim_edge = 128
            gnn_dim_node = 128
        elif (args.data == 'TALK'):
            gnn_dim_edge = 172
            gnn_dim_node = 172
        elif (args.data == 'STACK'):
            gnn_dim_edge = 172
            gnn_dim_node = 172
        elif (args.data == 'GDELT'):
            gnn_dim_edge = 182
            gnn_dim_node = 413
        elif (args.data == 'BITCOIN'):
            gnn_dim_edge = 172
            gnn_dim_node = 172
        elif (args.data == 'WIKI'):
            gnn_dim_edge = 0
            gnn_dim_node = 0
        else:
            raise RuntimeError("have not this dataset config!")
        

        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True


        from sampler.sampler_gpu import *
        use_gpu_sample = not args.use_cpu_sample

        no_neg = 'no_neg' in sample_param and sample_param['no_neg']
        


        if args.use_inductive:
            test_df = df[val_edge_end:]
            inductive_nodes = set(test_df.src.values).union(test_df.src.values)
            print("inductive nodes", len(inductive_nodes))
            neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
        else:
            neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])

        emb_buffer = None

        sampler_gpu = Sampler_GPU(g, sample_param['neighbor'], sample_param['layer'], emb_buffer)
        node_num = g['indptr'].shape[0] - 1
        edge_num = g['indices'].shape[0]

        if not (('no_sample' in sample_param and sample_param['no_sample']) or (use_gpu_sample)):
            sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                    sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                    sample_param['strategy']=='recent', sample_param['prop_time'],
                                    sample_param['history'], float(sample_param['duration']))
        else:
            sampler = None
        g = None
        del g
        emptyCache()



        if (gnn_dim_node == 0):
            node_feats = None
        if (gnn_dim_edge == 0):
            edge_feats = None
        prefetch_only_conn = prefetch_conn
        prefetch_conn = parent_conn
        


        model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, emb_buffer, combined=combine_first).cuda()

        mailbox = MailBox(memory_param, node_num, gnn_dim_edge, prefetch_conn=prefetch_conn) if memory_param['type'] != 'none' else None
        creterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])




        if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
            if node_feats is not None:
                node_feats = node_feats.cuda()
            if edge_feats is not None:
                edge_feats = edge_feats.cuda()
            if mailbox is not None:
                mailbox.move_to_gpu()

        



        if not os.path.isdir('models'):
            os.mkdir('models')
        if args.model_name == '':
            path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
        else:
            path_saver = 'models/{}.pkl'.format(args.model_name)
        best_ap = 0
        best_e = 0
        val_losses = list()




        from feat_buffer import *
        train_neg_sampler = None
        if (config.part_neg_sample):
            train_neg_sampler = TrainNegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])
        elif (hasattr(config, 'reuse_neg_sample') and config.reuse_neg_sample):
            train_neg_sampler = ReNegLinkSampler(node_num, args.reuse_ratio)
        else:
            train_neg_sampler = neg_link_sampler
            
        feat_buffer = Feat_buffer(args.data, None, datas, train_param, memory_param, train_edge_end, args.substream_size//train_param['batch_size'],\
                                sampler_gpu,train_neg_sampler, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge), node_num=node_num, edge_num = edge_num)


        test_ap, val_ap = [], []
        for e in range(train_param['epoch']):
            print('Epoch {:d}:'.format(e))
            time_sample = 0
            time_prep = 0
            time_tot = 0
            time_feat = 0
            time_model = 0
            time_opt = 0
            time_presample = 0
            time_gen_dgl = 0
            total_loss = 0
            time_per_batch = 0
            time_update_mem = 0
            time_update_mail = 0

            time_total_prep = 0
            time_total_strategy = 0
            time_total_compute = 0
            time_total_update = 0
            time_total_epoch = 0
            # training
            time_total_epoch_s = time.time()
            model.train()
            feat_buffer.mode = 'train'
            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                mailbox.set_buffer(feat_buffer)
                model.memory_updater.last_updated_nid = None
            if (feat_buffer is not None):
                feat_buffer.reset()
            

            sampleTime = 0
            startTime = time.time()

            sampler_gpu.mask_time = 0
            left, right = 0, 0
            batch_num = 0
            batch_size = train_param['batch_size']
            while True:
                right += batch_size
                right = min(train_edge_end, right)
                if (left >= right):
                    break

                src = datas['src'][left: right]
                dst = datas['dst'][left: right]
                times = datas['time'][left: right]
                eid = datas['eid'][left: right]

                loopTime = time.time()
                t_tot_s = time.time()
                time_presample_s = time.time()

                time_total_prep_s = time.time()
                feat_buffer.run_batch(batch_num)


                time_presample += time.time() - time_presample_s

                neg_start = (batch_num % feat_buffer.presample_batch) * train_param['batch_size']
                neg_end = min(feat_buffer.neg_sample_nodes.shape[0], ((batch_num % feat_buffer.presample_batch) + 1) * train_param['batch_size'])
                neg_sample_nodes = feat_buffer.neg_sample_nodes[neg_start: neg_end]

                root_nodes = np.concatenate([src, dst, neg_sample_nodes]).astype(np.int32)
                ts = np.concatenate([times, times, times]).astype(np.float32)
                
                t_sample_s = time.time()
                if (use_gpu_sample):
                    
                    root_nodes = torch.from_numpy(root_nodes).cuda()
                    root_ts = torch.from_numpy(ts).cuda()
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        ret = sampler_gpu.sample_layer(root_nodes[:pos_root_end], root_ts[:pos_root_end], cut_zombie=args.cut_zombie)
                    else:
                        ret = sampler_gpu.sample_layer(root_nodes, root_ts, cut_zombie=args.cut_zombie)
                else:
                    if sampler is not None:
                        if no_neg:
                            pos_root_end = root_nodes.shape[0] * 2 // 3
                            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                        else:
                            sampler.sample(root_nodes, ts)
                        ret = sampler.get_ret()
                        # time_sample += ret[0].sample_time()
                
                time_sample += time.time() - t_sample_s

                time1 = time.time()
                t_gen_dgl_s = time.time()
                t_prep_s = time.time()
                if (use_gpu_sample):
                    if gnn_param['arch'] != 'identity':
                        mfgs = sampler_gpu.gen_mfgs(ret)
                        root_nodes = root_nodes.cpu().numpy()
                    else:
                        mfgs = th_node_to_dgl_blocks(root_nodes, root_ts)  
                else:
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts)  
                time_gen_dgl += time.time() - t_gen_dgl_s
                time_feat_s = time.time()
                mfgs = prepare_input(mfgs, node_feats, edge_feats, feat_buffer = feat_buffer, combine_first=combine_first)
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                
                time_total_prep += time.time() - time_total_prep_s
                time_prep += time.time() - t_prep_s
                time_feat += time.time() - time_feat_s

                time_total_compute_s = time.time()
                optimizer.zero_grad()
                
                time1 = time.time()

                time_model_s = time.time()
                pred_pos, pred_neg = model(mfgs)
                time_model += time.time() - time_model_s

                time_opt_s = time.time()
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                total_loss += float(loss.item()) * train_param['batch_size']
                loss.backward()
                optimizer.step()
                time_opt += time.time() - time_opt_s
                time_total_compute += time.time() - time_total_compute_s
                t_prep_s = time.time()
                
                time_total_update_s = time.time()
                if mailbox is not None:
                    
                    
                    eid = eid.cuda()
                    
                    mem_edge_feats = feat_buffer.get_e_feat(eid) if edge_feats is not None else None
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                        block = sampler_gpu.gen_mfgs(ret, reverse=True)[0][0]
                        # block = mfgs[0][0]

                    time_upd_s = time.time()
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    time_update_mail += time.time() - time_upd_s

                    time_upd_s = time.time()
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                    time_update_mem += time.time() - time_upd_s

                time_prep += time.time() - t_prep_s
                time_tot += time.time() - t_tot_s
                time_total_update += time.time() - time_total_update_s
                # print(f"one loop time: {time.time() - loopTime:.4f}")

                time_per_batch += time.time() - t_tot_s

                left = right
                batch_num += 1

            print(f"total loop use time: {time.time() - startTime:.4f}")
            print(f"run batch{batch_num}total time: {time_tot:.2f}s,presample: {time_presample:.2f}s, sample: {time_sample:.2f}s, prep time: {time_prep:.2f}s, gen block: {time_gen_dgl:.2f}s, feat input: {time_feat:.2f}s, model run: {time_model:.2f}s,\
                loss and opt: {time_opt:.2f}s, update mem: {time_update_mem:.2f}s update mailbox: {time_update_mail:.2f}s")

            feat_buffer.mode = 'val'
            feat_buffer.refresh_memory()

            time_total_epoch += time.time() - time_total_epoch_s
            time_total_other = time_total_epoch - time_total_prep - time_total_strategy - time_total_compute - time_total_update
            print(f"prep:{time_total_prep:.4f}s strategy: {time_total_strategy:.4f}s compute: {time_total_compute:.4f}s update: {time_total_update:.4f}s epoch: {time_total_epoch:.4f}s other: {time_total_other:.4f}s")
            print(f"prep:{time_total_prep/time_total_epoch*100:.2f}% strategy: {time_total_strategy/time_total_epoch*100:.2f}% compute: {time_total_compute/time_total_epoch*100:.2f}% update: {time_total_update/time_total_epoch*100:.2f}% epoch: {time_total_epoch/time_total_epoch*100:.2f}% other: {time_total_other/time_total_epoch*100:.2f}%")


            if (not args.model_eval):
                continue
            eval_time_s = time.time()
            ap, auc = eval('val')
            
            if e > 2 and ap > best_ap:
                best_e = e
                best_ap = ap
                torch.save(model.state_dict(), path_saver)
            print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}, eval time: {:.2f}'.format(total_loss, ap, auc, time.time() - eval_time_s))
            val_ap.append(f'{ap:.6f}')
            # print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))

            # if (emb_buffer and emb_buffer.use_buffer):
            #     emb_buffer.reset_time()

            test_per_epoch = True
            if (test_per_epoch):
                if (args.model_eval):
                    model.eval()

                    if sampler is not None:
                        sampler.reset()
                    if mailbox is not None:
                        mailbox.reset()
                        model.memory_updater.last_updated_nid = None
                        eval('train')
                        eval('val')
                    ap, auc = eval('test')
                    # if args.eval_neg_samples > 1:
                    #     print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
                    # else:
                    #     print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
                    test_ap.append(f'{ap:.6f}')
            print(f'val: {val_ap}; test: {test_ap}')
        
        if (args.model_eval):
            print('Loading model at epoch {}...'.format(best_e))
            model.load_state_dict(torch.load(path_saver))
            model.eval()

            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                model.memory_updater.last_updated_nid = None
                eval('train')
                eval('val')
            ap, auc = eval('test')
            if args.eval_neg_samples > 1:
                print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
            else:
                print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
    except Exception as e:
        print(e)
    finally:
        parent_conn.send(('EXIT', ()))
        p.terminate()



