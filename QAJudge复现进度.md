### 00 参考文章和公开代码链接 

代码：

https://github.com/thunlp/QAJudge/

https://github.com/haoxizhong/pytorch-worker

文章（稍后发送PDF）：

https://ojs.aaai.org/index.php/AAAI/article/view/5479

### 1 进度和难点描述

目前作者公布的代码缺误已改完，难点是文章用于最终分类的Lightgbm模型是如何训练的、训练数据怎么构建文章和代码都没提。



lightgbm作用是作为文章”Question Net、Answer Net和Predict Net“架构中的Predict Net，输入是前两者的组合结果向量，输出标签类。



在代码中，先在sample_qajudge.config文件中说明lightgbm的pkl文件存放位置，在[QAJudge](https://github.com/thunlp/QAJudge/tree/393d8cafac090c1161157d25080c6b73713676a9)/[model](https://github.com/thunlp/QAJudge/tree/393d8cafac090c1161157d25080c6b73713676a9/model)/[zm_predict](https://github.com/thunlp/QAJudge/tree/393d8cafac090c1161157d25080c6b73713676a9/model/zm_predict)/**dqn.py** 模型代码中引用，输入强化学习输出的final_state向量，输出最终结果标签类。用到这个模型的目前只发现50-55行，和90行。


50-55
        lgb_path = config.get("ml", "lgb_path")
        self.lgb = pickle.load(open(os.path.join(lgb_path, "predict_net.pkl"), "rb"))

    def output_model(self, state_list):
        x = self.lgb.predict(state_list).tolist()
        return x

90    final_result = self.output_model([final_state])[0]



