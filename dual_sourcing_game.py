import numpy as np
import matplotlib.pyplot as plt

class DualSourcing():
    #config=dictionary of {"regular_leadtime":XXX,...}
    #y is the discount factor
    #demand=a list of len(episode_len*episode_rep), and initializes to poisson distribution
    def __init__(self, config,episode_len=1000,episode_rep=50,demand=None):
        self.rl = config['regular_leadtime']
        self.el = config['express_leadtime']
        self.rc = config['regular_cost']
        self.ec = config['express_cost']
        self.sc = config['store_cost']
        self.bc = config['back_cost']
        self.max_order=config['max_order']
        self.max_inventory=config['max_inventory']
        self.y=config['y']
        self.starting_state=config['starting_state']
        self.episode_len=episode_len
        self.episode_rep=episode_rep
        self.total_episode_len=self.episode_len*self.episode_rep
        if demand==None:
            #simulate an iid demand with constant+poisson random variable
            self.poisson_lambda=6
            self.fixed_demand=2
            demand=np.random.poisson(self.poisson_lambda,self.total_episode_len)
            demand=[d+self.fixed_demand for d in demand]
        if len(demand)!=self.total_episode_len:
            raise ValueError("Length of Demand {} must match episode length {}".format(len(demand),self.total_episode_len))
        self.demand=demand
#         state=[cur_inventory,[reg_order_by_rl,reg_order_by_rl-1,reg_order_by_rl-2...,reg_order_just_made],
#                [exp_order_by_el,exp_order_by_el-1...exp_order_just_made]]
        self.state = self.starting_state
        self.inventory_overflow=0
        self.inventory_underflow=0
        self.inventory_history=[]
        self.reward_history=[]
        self.demand_history=[]
        self.reg_order_history=[]
        self.exp_order_history=[]
        self.time_stamp=0
    # Auxilary function 
    #update inventory and pipeline vectors
    def update_inventory(self,demand,r_order,e_order):
        cur_inventory=self.state[0]+self.state[1][0]+self.state[2][0]-demand
        if cur_inventory>self.max_inventory:
            self.inventory_overflow=cur_inventory-self.max_inventory
            cur_inventory=self.max_inventory
        else:
            self.inventory_overflow=0
        if cur_inventory<-self.max_inventory:
            self.inventory_underflow=-self.max_inventory-cur_inventory
            cur_inventory=-self.max_inventory
        else:
            self.inventory_underflow=0
        self.state[0]=cur_inventory
        self.state[1]=self.state[1][1:]+[r_order]
        self.state[2]=self.state[2][1:]+[e_order]
        return self.state[0]
    #compute reward, the negative of cost
    #we modified the reward term to penalize 
    #i)exceeding max stock or dropping below min stock
    #ii)ordering much when stock is already high; constantly pushing the stock up
    #iii)ordering little when stock is negative
    #in other words, we would like to encourage the agent to push inventory towards 0, but never falls below 0 
    def calcuate_reward(self,r_order,e_order):
        cost=(self.rc*r_order+self.ec*e_order+self.sc*max(self.state[0],0)+self.bc*max(-self.state[0],0))
        stocking_penality=0
        insufficiency_penalty=0
        overflow_penalty=0
        underflow_penalty=0
        past_demand=self.demand_history[self.time_stamp-1]
        if self.time_stamp>1 and self.state[0]>(self.el)*past_demand:
            stocking_penality=2.4*self.sc*(e_order)
        if self.state[0]+self.state[1][0]+self.state[2][0]<0:
            insufficiency_penalty=5*self.bc*(-self.state[0]-r_order-e_order)
        if self.inventory_overflow>0:
            overflow_penalty=8.2*self.sc*(self.inventory_overflow+r_order+e_order)
        if self.inventory_underflow>0:
            underflow_penalty=6.4*self.bc*(self.inventory_underflow-r_order-e_order)
        return -(cost+stocking_penality+insufficiency_penalty+overflow_penalty+underflow_penalty)
    def reset(self):
        self.state = np.asarray(self.starting_state)
        self.inventory_history=[]
        self.reward_history=[]
        self.demand_history=[]
        self.reg_order_history=[]
        self.exp_order_history=[]
        self.time_stamp=0
        return self.state 
    def plot(self,time_start,time_end):
        fig=plt.figure(figsize=[12,10])
        ax_inv=fig.add_subplot(3,1,1)
        ax_inv.set_xlabel("TimeStamp")
        ax_inv.set_ylabel("Inventory Level")
        ax_demand=fig.add_subplot(3,1,2)
        ax_demand.set_xlabel("TimeStamp")
        ax_demand.set_ylabel("Demand Level")
        ax_orders=fig.add_subplot(3,1,3)
        ax_orders.set_xlabel("TimeStamp")
        ax_orders.set_ylabel("Order Quantity")
        ax_inv.plot(list(range(time_start%self.episode_len,time_end%self.episode_len+1)),self.inventory_history[time_start:time_end+1])
        ax_demand.plot(list(range(time_start%self.episode_len,time_end%self.episode_len+1)),self.demand_history[time_start:time_end+1])
        ax_orders.plot(list(range(time_start%self.episode_len,time_end%self.episode_len+1)),self.reg_order_history[time_start:time_end+1],label="Regular Orders")
        ax_orders.plot(list(range(time_start%self.episode_len,time_end%self.episode_len+1)),self.exp_order_history[time_start:time_end+1],label="Express Orders")
        ax_orders.legend()
        plt.show()
    #this function allows users to play the game
    def play(self):
        while(self.time_stamp<self.episode_len):
            cur_demand=self.demand[self.time_stamp]
            self.time_stamp+=1
            print("Current Demand is "+str(cur_demand))
            print("Your current inventory level is "+str(self.state[0]))
            print("Your Pipeline vector from regular supplier is"+str(self.state[1]))
            print("Your Pipeline vector from express supplier is"+str(self.state[2]))
            print("Input your regular and express orders,separated by comma")
            str_inp=str(input()).split(",")
            order_reg=int(str_inp[0])
            order_exp=int(str_inp[1])
            self.inventory_history.append(self.update_inventory(cur_demand,order_reg,order_exp))
            self.reward_history.append(self.calcuate_reward(order_reg,order_exp))
            self.demand_history.append(cur_demand)
            self.reg_order_history.append(order_reg)
            self.exp_order_history.append(order_exp)
            fig=plt.figure(figsize=[12,10])
            ax_inv=fig.add_subplot(2,2,1)
            ax_inv.set_xlabel("TimeStamp")
            ax_inv.set_ylabel("Inventory Level")
            ax_reward=fig.add_subplot(2,2,2)
            ax_reward.set_xlabel("TimeStamp")
            ax_reward.set_ylabel("Average Reward Level")
            ax_demand=fig.add_subplot(2,2,3)
            ax_demand.set_xlabel("TimeStamp")
            ax_demand.set_ylabel("Demand Level")
            ax_orders=fig.add_subplot(2,2,4)
            ax_orders.set_xlabel("TimeStamp")
            ax_orders.set_ylabel("Order Quantity")
            ax_inv.plot(list(range(self.time_stamp)),self.inventory_history)
            ax_reward.plot(list(range(self.time_stamp)),[val/(index+1) for index,val in enumerate(self.reward_history)])
            ax_demand.plot(list(range(self.time_stamp)),self.demand_history)
            ax_orders.plot(list(range(self.time_stamp)),self.reg_order_history,label="Regular Orders")
            ax_orders.plot(list(range(self.time_stamp)),self.exp_order_history,label="Express Orders")
            ax_orders.legend()
            plt.show()
    def step(self,reg_order,exp_order):
            if self.time_stamp==self.total_episode_len:
                raise ValueError("Time stamp {} must be less or equal to length of episode".format(self.time_stamp+1,self.episode_len))
            cur_demand=self.demand[self.time_stamp]
            self.time_stamp+=1
            self.demand_history.append(cur_demand)
            self.inventory_history.append(self.update_inventory(cur_demand,reg_order,exp_order))
            self.reward=self.calcuate_reward(reg_order,exp_order)
            self.reward_history.append(self.reward)
            self.reg_order_history.append(reg_order)
            self.exp_order_history.append(exp_order)
            if self.time_stamp==self.total_episode_len:
                fig=plt.figure(figsize=[12,10])
                ax_inv=fig.add_subplot(2,2,1)
                ax_inv.set_xlabel("TimeStamp")
                ax_inv.set_ylabel("Inventory Level(last episode)")
                ax_reward=fig.add_subplot(2,2,2)
                ax_reward.set_xlabel("Episode Number")
                ax_reward.set_ylabel("Average Reward Level")
                ax_demand=fig.add_subplot(2,2,3)
                ax_demand.set_xlabel("TimeStamp")
                ax_demand.set_ylabel("Demand Level(last episode)")
                ax_orders=fig.add_subplot(2,2,4)
                ax_orders.set_xlabel("TimeStamp")
                ax_orders.set_ylabel("Order Quantity(last episode)")
                ax_inv.plot(list(range(self.time_stamp-self.episode_len,self.time_stamp)),self.inventory_history[self.time_stamp-self.episode_len:])
                reward_matrix=np.reshape(self.reward_history,(self.episode_rep,self.episode_len))
                avg_rewards=np.mean(reward_matrix,axis=1)
                ax_reward.plot(avg_rewards)
                ax_demand.plot(list(range(self.time_stamp-self.episode_len,self.time_stamp)),self.demand_history[self.time_stamp-self.episode_len:])
                ax_orders.plot(list(range(self.time_stamp-self.episode_len,self.time_stamp)),self.reg_order_history[self.time_stamp-self.episode_len:],label="Regular Orders")
                ax_orders.plot(list(range(self.time_stamp-self.episode_len,self.time_stamp)),self.exp_order_history[self.time_stamp-self.episode_len:],label="Express Orders")
                ax_orders.legend()
                plt.show()
                print("Info on Last Episode:")
                print("Avg Inventory Stock")
                avg_inventory=sum(self.inventory_history[self.time_stamp-self.episode_len:])/self.episode_len
                print(avg_inventory)
                print("Variance of Inventory Stock")
                var_inventory = sum((x - avg_inventory) ** 2 for x in self.inventory_history[self.time_stamp-self.episode_len:]) /self.episode_len
                print(var_inventory)
                print("Avg Inventory Holding(多出来的部分)")
                total_overflow=sum(max(0,self.inventory_history[i]) for i in range(self.time_stamp-self.episode_len,self.time_stamp))
                print(total_overflow/self.episode_len)
                print("Avg Inventory Backloss")
                total_backloss=sum(-min(0,self.inventory_history[i]) for i in range(self.time_stamp-self.episode_len,self.time_stamp))
                print(total_backloss/self.episode_len)
                print("Avg Cost per day")
                total_cost=total_overflow*self.sc+total_backloss*self.bc+sum(self.reg_order_history[self.time_stamp-self.episode_len:])*self.rc+sum(self.exp_order_history[self.time_stamp-self.episode_len:])*self.ec
                print(total_cost/self.episode_len)

# In[47]:


# config={'regular_leadtime':6,'express_leadtime':2,'regular_cost':10,'express_cost':20,"max_order":20,
#        'store_cost':4,'back_cost':35,'y':0.9,'starting_state':[20,[0 for _ in range(6)],[0 for _ in range(2)]]}
# game=DualSourcing(config,episode_len=20)
# game.play()


# In[ ]:




