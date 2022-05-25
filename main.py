import matplotlib.pyplot as plt
import numpy as np
import torch
from algs.ql import Qlearning
from algs.rql import RQlearning
from algs.deep_rql import DeepRQL
from envs.cloud import CloudEnv
from envs.spelunking import SpelunkingEnv
from envs.palindrome import PalindromeEnv
from envs.utils import RMDPasMDP, RMDPFlatten

#
# Run experiments
#

num_runs = 10 # Number of trials to run
skip_to_plotting = False # Use existing data to plot and skip experiments
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

if not skip_to_plotting:
    # Cloud (RQL)
    for n in range(num_runs):
        env = CloudEnv()
        learner = RQlearning(env, alpha = 0.02, epsilon = 0.1, episode_limit = 200, discretize_step=0.001)
        log = learner.train(train_steps=100000, test_num=100, test_freq=100)
        np.save('./results/cloud/rql_'+str(n)+'.npy', np.array(log))

    # Cloud (QL + RMDP)
    for n in range(num_runs):
        env = RMDPasMDP(CloudEnv)
        learner = Qlearning(env, alpha = 0.02, epsilon = 0.1, episode_limit = 200)
        log = learner.train(train_steps=100000, test_num=100, test_freq=100)
        np.save('./results/cloud/ql_'+str(n)+'.npy', np.array(log))

    # Spelunking (RQL)
    for n in range(num_runs):
        env = SpelunkingEnv()
        learner = RQlearning(env, alpha = 0.2, epsilon = 0.1, episode_limit = 200)
        log = learner.train(train_steps=200000, test_num=100)
        np.save('./results/spelunking/rql_'+str(n)+'.npy', np.array(log))

    # Spelunking (QL + RMDP)
    for n in range(num_runs):
        env = RMDPasMDP(SpelunkingEnv)
        learner = Qlearning(env, alpha = 0.2, epsilon = 0.1, episode_limit = 200)
        log = learner.train(train_steps=200000, test_num=100)
        np.save('./results/spelunking/ql_'+str(n)+'.npy', np.array(log))

    # Palindrome (Deep RQL)
    for n in range(num_runs):
        env = PalindromeEnv(encoding='vector')
        deepRQL = DeepRQL()

        net, logs = deepRQL.train(env,
                                epsilon=1, \
                                final_epsilon=0.1, \
                                epsilon_frac=0.1, \
                                buffer_size=int(20e3), \
                                num_train_steps=int(300e3), \
                                hidden_layers=2, \
                                hidden_dim=128, \
                                target_update_freq=500, \
                                batch_size=256, \
                                lr=0.0005, \
                                episode_limit=-1, \
                                log_freq=1000, \
                                test_num=100, \
                                device=device)
        
        torch.save(net.state_dict(), './results/palindrome/rql_'+str(n)+'.pt')
        np.save('./results/palindrome/rql_'+str(n)+'.npy', np.array(logs))

    # Palindrome (DQN + RMDP)
    for n in range(num_runs):
        env = RMDPFlatten(PalindromeEnv, encoding='vector')
        deepRQL = DeepRQL()

        net, logs = deepRQL.train(env,
                                epsilon=1, \
                                final_epsilon=0.1, \
                                epsilon_frac=0.1, \
                                buffer_size=int(20e3), \
                                num_train_steps=int(300e3), \
                                hidden_layers=2, \
                                hidden_dim=128, \
                                target_update_freq=500, \
                                batch_size=256, \
                                lr=0.0005, \
                                episode_limit=-1, \
                                log_freq=1000, \
                                test_num=100, \
                                device=device)
        
        torch.save(net.state_dict(), './results/palindrome/ql_'+str(n)+'.pt')
        np.save('./results/palindrome/ql_'+str(n)+'.npy', np.array(logs))


#
# Plot experiments
#

# Cloud
rql = []
ql = []
for n in range(num_runs):
    arr = np.load('./results/cloud/rql_'+str(n)+'.npy')
    rql.append(arr)
    arr = np.load('./results/cloud/ql_'+str(n)+'.npy')
    ql.append(arr)
rql = np.array(rql)
ql = np.array(ql)
    
rql_perc = np.percentile(rql, [10,90], axis=0)
plt.fill_between(rql[0,:,0], rql_perc[0,:,1], rql_perc[1,:,1], alpha=0.2)
plt.plot(rql[0,:,0], rql.mean(axis=0)[:,1])
ql_perc = np.percentile(ql, [10,90], axis=0)
plt.fill_between(ql[0,:,0], ql_perc[0,:,1], ql_perc[1,:,1], alpha=0.2)
plt.plot(ql[0,:,0], ql.mean(axis=0)[:,1])
plt.legend(('RQL (ours)', 'QL + RMDP'))
plt.title('Cloud')
plt.xlabel('Training step')
plt.ylabel('Total reward')
plt.xticks(np.arange(100001, step=25000),[0, '2.5e4', '5e4', '7.5e4', '1e5'])
plt.rc('font', family='serif', size=16)
plt.xlim([0,50000])
plt.ylim([-6,-5])
plt.savefig('./results/cloud_curves.pdf', bbox_inches='tight')
plt.close()

# Spelunking
rql = []
ql = []
for n in range(num_runs):
    arr = np.load('./results/spelunking/rql_'+str(n)+'.npy')
    rql.append(arr)
    arr = np.load('./results/spelunking/ql_'+str(n)+'.npy')
    ql.append(arr)
rql = np.array(rql)
ql = np.array(ql)
    
rql_perc = np.percentile(rql, [10,90], axis=0)
plt.fill_between(rql[0,:,0], rql_perc[0,:,1], rql_perc[1,:,1], alpha=0.2)
plt.plot(rql[0,:,0], rql.mean(axis=0)[:,1])
ql_perc = np.percentile(ql, [10,90], axis=0)
plt.fill_between(ql[0,:,0], ql_perc[0,:,1], ql_perc[1,:,1], alpha=0.2)
plt.plot(ql[0,:,0], ql.mean(axis=0)[:,1])
plt.legend(('RQL (ours)', 'QL + RMDP'))
plt.title('Spelunking')
plt.xlabel('Training step')
# plt.ylabel('Total reward')
plt.xticks(np.arange(200001, step=100000), ['0', '1e5', '2e5'])
plt.rc('font', family='serif', size=16)
plt.savefig('./results/spelunking_curves.pdf', bbox_inches='tight')
plt.close()

# Palindrome
rql = []
ql = []
for n in range(num_runs):
    arr = np.load('./results/palindrome/rql_'+str(n)+'.npy')
    rql.append(arr)
    arr = np.load('./results/palindrome/ql_'+str(n)+'.npy')
    ql.append(arr)
rql = np.array(rql)
ql = np.array(ql)
    
rql_perc = np.percentile(rql, [10,90], axis=0)
plt.fill_between(rql[0,:,0], rql_perc[0,:,1], rql_perc[1,:,1], alpha=0.2)
plt.plot(rql[0,:,0], rql.mean(axis=0)[:,1])
ql_perc = np.percentile(ql, [10,90], axis=0)
plt.fill_between(ql[0,:,0], ql_perc[0,:,1], ql_perc[1,:,1], alpha=0.2)
plt.plot(ql[0,:,0], ql.mean(axis=0)[:,1])
plt.legend(('Deep RQL (ours)', 'DQN + RMDP'))
plt.xlabel('Training step')
# plt.ylabel('Total reward')
plt.xticks(np.arange(300001, step=100000), ['0', '1e5', '2e5', '3e5'])
plt.title('Palindrome')
plt.rc('font', family='serif', size=16)
plt.savefig('./results/palindrome_curves.pdf', bbox_inches='tight')
plt.close()