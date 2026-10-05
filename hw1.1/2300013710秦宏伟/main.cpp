#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "tictactoe.hpp"

using namespace std;

class TicTacToePolicyBase
{
public:
    virtual TicTacToe::Action operator()(const TicTacToe::State &state) const = 0;
    virtual void update(const TicTacToe::State &s_now, float r,
                        const TicTacToe::State &s_next, bool terminal) = 0; // ← 加这个
};

// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase
{
public:
    TicTacToe::Action operator()(const TicTacToe::State &state) const
    {
        vector<TicTacToe::Action> actions = state.action_space();
        int n_action = actions.size();
        int action_id = rand() % n_action;
        if (state.turn == TicTacToe::PLAYER_X)
        {
            return actions[action_id];
        }
        else
        {
            return actions[action_id];
        }
    }
    TicTacToePolicyRandom()
    {
        srand(time(nullptr));
    }
    void update(const TicTacToe::State &, float,
                const TicTacToe::State &, bool) override {}
};

// select the first valid action.
class TicTacToePolicyDefault : public TicTacToePolicyBase
{
public:
    TicTacToe::Action operator()(const TicTacToe::State &state) const
    {
        vector<TicTacToe::Action> actions = state.action_space();
        if (state.turn == TicTacToe::PLAYER_X)
        {
            // TODO

            return actions[0];
        }
        else
        {
            return actions[0];
        }
    }
    TicTacToePolicyDefault() {}
    void update(const TicTacToe::State &, float,
                const TicTacToe::State &, bool) override {}
};

// my policy, select the best action based on the value function
class TicTacToePolicyEx : public TicTacToePolicyBase
{
    // 需要定义
private:
    float learning_rate = 0.1f;        // 学习率，表示每次更新价值函数时，当前状态的价值会向目标状态的价值靠近多少
    float discount_factor = 0.95f;     // 折扣因子，表示未来奖励的折扣程度
    unordered_map<int, float> v_table; // board -> value， board本身是用int表示的，value是float类型的，我们存储的是X胜利的概率，对于X实际赢的概率，v_table的值是1，对于O实际赢的概率，v_table的值是0，对于平局的概率，v_table的值是0.5，对于未结束的游戏，v_table的值是0.5
    float epsilon = 0.1f;              // epsilon-greedy策略的epsilon值，表示有多少概率选择随机动作，而不是选择最优动作
    float get_v(int board_key) const
    {
        auto it = v_table.find(board_key);
        return (it == v_table.end()) ? 0.5f : it->second;
    }

public:
    TicTacToe::Action operator()(const TicTacToe::State &state) const
    {
        // 假定当前状态是X的回合，选择一个最优动作
        vector<TicTacToe::Action> actions = state.action_space();
        TicTacToe::State best_next = state;
        TicTacToe::Action best_action = actions[0];
        best_next.put(actions[0]);
        float best_value = get_v(best_next.board);
        // 进入实际的策略选择逻辑
        if (state.turn == TicTacToe::PLAYER_X)
        {
            // 使用epsilon-greedy策略
            // recap: rand()返回一个0到RAND_MAX之间的随机整数，rand() % 100返回一个0到99之间的随机整数，epsilon * 100是一个0到100之间的浮点数，表示选择随机动作的概率
            if (rand() % 100 < epsilon * 100)
            {
                return actions[rand() % actions.size()];
            }
            // TODO
            // 根据当前状态和价值函数，选择一个最优动作
            for (const auto &action : actions)
            {
                TicTacToe::State next_state = state;
                next_state.put(action);
                int board_key = next_state.board;
                float value = get_v(board_key); // 得到当前动作对应的下一个状态的价值
                if (value > best_value)
                {
                    best_value = value;
                    best_action = action;
                }
            }

            return best_action;
        }
        else
        {
            // 使用epsilon-greedy策略
            if (rand() % 100 < epsilon * 100)
            {
                return actions[rand() % actions.size()];
            }
            // 假定当前状态是O的回合，选择一个最优动作
            for (const auto &action : actions)
            {
                TicTacToe::State next_state = state;
                next_state.put(action);
                int board_key = next_state.board;
                float value = -1 * get_v(board_key); // 得到当前动作对应的下一个状态的价值
                if (value < best_value)
                {
                    best_value = value;
                    best_action = action;
                }
            }
            return best_action;
        }
    }
    TicTacToePolicyEx()
    {
        this->epsilon = 0.1;
        this->learning_rate = 0.1;
        this->discount_factor = 0.95;
    }
    void set_epsilon(float value)
    {
        epsilon = value;
    }
    // 更新价值函数
    void update(const TicTacToe::State &s_now, float reward,
                const TicTacToe::State &s_next, bool terminal) override
    {
        // 逻辑是
        // 1. 获取当前状态的价值
        // 2. 计算目标价值，如果是终局状态，目标价值就是奖励，否则就是奖励加上折扣因子乘以下一个状态的价值
        // 3. 更新当前状态的价值，使用学习率来控制更新的幅度
        int key = s_now.board;
        float old_v = get_v(key);
        float target = terminal ? reward
                                : reward + discount_factor * get_v(s_next.board);
        v_table[key] = old_v + learning_rate * (target - old_v);
    }
};

#include <chrono>
#include <thread>

// randomly select action
int main()
{
    // 训练阶段
    TicTacToePolicyEx ex_policy;
    TicTacToePolicyDefault default_policy;

    TicTacToePolicyBase &ref_x = ex_policy;
    TicTacToePolicyBase &ref_o = default_policy;

    TicTacToe env(false);
    env.reset();

    for (int episode = 0; episode < 100000; ++episode)
    {
        env.reset();
        while (true)
        {
            // 只学习 X 的决策状态；O 始终使用固定的默认策略。
            TicTacToe::State x_state = env.get_state();
            env.step(ref_x(x_state));

            if (env.done())
            {
                int w = env.winner();
                float reward = w == TicTacToe::PLAYER_X ? 1.0f : 0.5f;
                ref_x.update(x_state, reward, env.get_state(), true);
                break;
            }

            TicTacToe::State o_state = env.get_state();
            env.step(ref_o(o_state));

            if (env.done())
            {
                int w = env.winner();
                float reward = w == TicTacToe::PLAYER_O ? 0.0f : 0.5f;
                ref_x.update(x_state, reward, env.get_state(), true);
                break;
            }

            ref_x.update(x_state, 0.0f, env.get_state(), false);
        }
    }
    cout << "===================================" << endl;
    // 测试阶段
    bool done = false;
    // set verbose true
    ex_policy.set_epsilon(0.0f);
    env.verbose = true;
    env.reset(); // 重置环境
    while (not done)
    {
        TicTacToe::State state = env.get_state();
        // X 用训练好的策略，O 用默认策略
        TicTacToePolicyBase &cur = (state.turn == TicTacToe::PLAYER_X) ? ref_x : ref_o;
        TicTacToe::Action action = cur(state);
        env.step(action);
        done = env.done();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    return 0;
};