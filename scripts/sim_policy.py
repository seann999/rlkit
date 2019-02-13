import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger

from rlkit.envs.wrappers import NormalizedBoxEnv
from custom_env import create_swingup

filename = str(uuid.uuid4())

import numpy as np

from rlkit.policies.base import Policy

import torch
import torch.nn.functional as F


class OptionPolicy(Policy):
    def __init__(self, policy, skill_dim, auto=False):
        self.policy = policy
        self.skill_dim = skill_dim
        
        self.steps = 1
        self.current_z = 0
        
        self.set_z(self.current_z)
        
    def set_z(self, z_index):
        vec = np.zeros(self.skill_dim)
        vec[min(len(vec)-1, z_index)] = 1
        self.skill_vec = vec

    def get_action(self, observation):
        observation = np.concatenate([observation, self.skill_vec])
        
        self.steps += 1
        if self.steps % 100 == 0:
            self.current_z += 1
            self.set_z(self.current_z)
        
        return self.policy.get_action(observation)


def simulate_policy(args):
    data = joblib.load(args.file)
    
    cont = False
    
    if 'policies' in data:
        policy = data['policies'][0]
    else:
        policy = data['policy']
    env = NormalizedBoxEnv(create_swingup())#data['env']
    
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
        data['qf1'].cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)

    diayn = 'df' in data
    rnd = 'rf' in data
    
    if diayn:
        skills = len(data['eval_policy'].skill_vec)
        disc = data['df']
        
        policy = OptionPolicy(policy, skills, cont)
        if args.gpu:
            disc.cuda()
        if isinstance(policy, PyTorchModule):
            disc.train(False)
            
    if rnd:
        data['rf'].cuda()
        data['pf'].cuda()
        data['qf1'].cuda()

    import cv2
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*"H264"), 30, (640, 480))
    index = 0
    
    truth, pred = [], []
    
    if cont:
        eps = 1
    elif diayn:
        eps = skills * 2 
    else:
        eps = 5
        
    Rs = []

    for ep in range(eps):
        if diayn and not cont:
            z_index = ep // 2
            policy.set_z(z_index)
        
        path = rollout(
            env,
            policy,
            max_path_length=args.H * skills if cont else args.H,
            animated=True,
        )
        
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
        
        total_r = 0
        
        if diayn:
            predictions = F.log_softmax(disc(torch.FloatTensor(path['observations']).cuda()), 1).cpu().detach().numpy()
            probs = predictions.max(1)
            labels = predictions.argmax(1)
        
            if cont:
                for k in range(skills):
                    truth.extend([k] * 100)
            else:
                truth.extend([z_index] * len(labels))
            pred.extend(labels.tolist())
            
        if rnd:
            random_feats = data['rf'](torch.FloatTensor(path['observations']).cuda())
            pred_feats = data['pf'](torch.FloatTensor(path['observations']).cuda())
            
            i_rewards = ((random_feats - pred_feats)**2.0).sum(1).cpu().data.numpy()
        
        q_pred = data['qf1'](torch.FloatTensor(path['observations']).cuda(), torch.FloatTensor(path['actions']).cuda()).cpu().data.numpy()

        for i, (img, r, s) in enumerate(zip(path['images'], path['rewards'], path['observations'])):
            #video.write(img[:,:,::-1].astype(np.uint8))
            total_r += r[0]
            img = img.copy()
            img = np.rot90(img, 3).copy()
            col = (255, 0, 255)
            cv2.putText(img, "step: %d" % (i+1), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA) 
            
            if diayn:
                if cont:
                    cv2.putText(img, "z: %s" % str(truth[i]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, "z: %s" % str(z_index), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    
                cv2.putText(img, "disc_pred: %s (%.3f)" % (labels[i], probs[i]), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA) 
                cv2.putText(img, "reward: %.3f" % r[0], (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA) 
                cv2.putText(img, "total reward: %.1f" % total_r, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "action: %s" % path['actions'][i], (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "reward: %.1f" % r[0], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA) 
                cv2.putText(img, "total reward: %.1f" % total_r, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
                y = 120
            
            if rnd:
                cv2.putText(img, "i reward (unscaled): %.3f" % i_rewards[i], (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA) 
                #cv2.rectangle(img, (20, 180), (20 + int(q_pred[i, 0]), 200), (255, 0, 255), -1)
                cv2.rectangle(img, (20, 200), (20 + int(i_rewards[i] * 10), 220), (255, 255, 0), -1)
                y = 220
                
            try:
                y += 40
                cv2.putText(img, "Q: %.3f" % q_pred[i], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA) 
            except:
                y += 40
                cv2.putText(img, "Q:" + str([q for q in q_pred[i]]), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA) 
            y += 40
            cv2.putText(img, str(["%.3f" % x for x in path['observations'][i]]), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA) 
            
            try:
                cv2.imwrite("frames/%06d.png" % index, img[:,:,::-1])
            except:
                cv2.imwrite("frames/%06d.png" % index, img[:,:])
            index += 1
            
        if diayn:
            print(z_index, ":", total_r)
        Rs.append(total_r)
        
    print("best", np.argmax(Rs))
    print("worst", np.argmin(Rs))
            
    video.release()
    print("wrote video")
    
    if diayn:
        import sklearn
        from sklearn.metrics import confusion_matrix
        import matplotlib as mpl
        import itertools
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        normalize = False
        classes = range(skills)
        cm = confusion_matrix(truth, pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(skills)
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        """
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        """

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig("confusion.png")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
