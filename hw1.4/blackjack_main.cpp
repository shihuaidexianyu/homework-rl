// ============================================================================
// blackjack_main.cpp —— 21 点的策略与训练主程序（智能体层）
//
// 对应 Sutton《强化学习》Example 5.1 / 5.3：
//   用"带探索起点的蒙特卡洛控制"（MC ES）求玩家对庄家固定策略（≥17 停牌）
//   的最优对策，最终输出可对照书上 P100 Figure 5.2 的策略表。
//
// 结构：
//   BlackjackPolicyBase             策略抽象接口
//   BlackjackPolicyDefault          写死的演示策略（玩家≥20停，庄家≥17停）
//   BlackjackPolicyLearnableDefault 可学习策略（作业主体：实现 update_value / update_policy）
//   main()                          当前为演示循环，作业需改为"训练 + 打印策略"
// ============================================================================
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <utility>
#include <vector>
#include "blackjack.hpp"

// ---- 策略抽象基类：给出一个状态，返回一个动作（HIT/STICK） ----
class BlackjackPolicyBase
{
public:
    virtual int operator()(const Blackjack::State &state) = 0;
};

// ---- 固定策略（演示用）：玩家点数 ≥20 停牌否则要牌；庄家 ≥17 停牌否则要牌 ----
class BlackjackPolicyDefault : public BlackjackPolicyBase
{
public:
    int operator()(const Blackjack::State &state)
    {
        if (state.turn == Blackjack::PLAYER)
        {
            return state.player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
        }
        else
        {
            return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
        }
    }
};

// ---- 可学习策略（作业主体）：庄家仍用固定规则，玩家动作查 policy 表 ----
// 学习目标是蒙特卡洛 ES 策略迭代：
//   update_value  —— 策略评估：MC 估计当前策略下的动作价值 Q(s,a)
//   update_policy —— 策略改进：π(s) ← argmax_a Q(s,a)
class BlackjackPolicyLearnableDefault : public BlackjackPolicyBase
{
    static constexpr const char *ACTION_NAME = "HS"; // 打印用：HIT→'H'，STICK→'S'

public:
    // 决策接口：庄家固定（≥17 停牌），玩家查当前策略表
    int operator()(const Blackjack::State &state)
    {
        if (state.turn == Blackjack::DEALER)
        {
            return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
        }
        else
        {
            return policy[state.dealer_shown][state.player_ace][state.player_sum];
        }
    }
    // ---- 【待实现】策略改进：对每个状态取使 Q 最大的动作作为新策略 ----
    void update_policy()
    {
        // TODO

        // simply take argmax
        // 即：policy[s] = (value[s][STICK] > value[s][HIT]) ? STICK : HIT
        for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
        {
            for (int player_ace = 0; player_ace <= 1; ++player_ace)
            {
                for (int player_sum = 11; player_sum <= 20; ++player_sum)
                {
                    // 枚举所有状态 (dealer_shown, player_ace, player_sum)，取使 Q 最大的动作作为新策略
                    int action = policy[dealer_shown][player_ace][player_sum];
                    if (value[dealer_shown][player_ace][player_sum][Blackjack::STICK] > value[dealer_shown][player_ace][player_sum][Blackjack::HIT])
                    {
                        action = Blackjack::STICK;
                    }
                    else
                    {
                        action = Blackjack::HIT;
                    }
                    policy[dealer_shown][player_ace][player_sum] = action;
                }
            }
        }
    }
    // ---- 【待实现】策略评估：蒙特卡洛估计动作价值 ----
    // n iterations for each start (initial state and first action), no need to modify.
    // 对每个"起点"（初始状态 × 首个动作）各模拟 n 局
    void update_value(Blackjack &env, int n = 10000)
    {
        // TODO

        // REMEMBER to call set_value_initial() at the beginning
        // 开头记得调用 set_value_initial() 清零 value 和计数
        // simulate from every possible initial state:(dealer_shown, player_ace, player_sum) \
                (call Blackjack::reset(,,) to do this) and player's every possible first action
        // 用 Blackjack::reset(dealer_shown, player_ace, player_sum) 枚举所有初始状态，
        // 并让玩家的首个动作也遍历 HIT / STICK（探索起点 Exploring Starts）
        // BE AWARE only use player's steps (rather than dealer's) to update value estimation.
        // 注意：只用"玩家的步"更新价值；庄家的步属于环境动态，不进 episode
        // 更新方式：首次访问 MC，增量平均
        //   count[s][a] += 1; value[s][a] += (G - value[s][a]) / count[s][a]
        set_value_initial(); // 清零 value 与计数
        for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
        {
            for (int player_ace = 0; player_ace <= 1; ++player_ace)
            {
                for (int player_sum = 11; player_sum <= 20; ++player_sum)
                {

                    for (int first_action = Blackjack::HIT; first_action <= Blackjack::STICK; ++first_action)
                    {
                        for (int i = 0; i < n; ++i)
                        {
                            // 每个起点各模拟 n 局
                            episode.clear();                                 // 记录玩家的轨迹（状态+动作）
                            env.reset(dealer_shown, player_ace, player_sum); // 固定初始状态
                            // 遍历起点动作（HIT / STICK）
                            episode.emplace_back(env.state(), first_action); // 记录玩家的首步
                            Blackjack::StepResult result = env.step(first_action);
                            while (not result.done and env.state().turn == Blackjack::PLAYER)
                            {
                                int action = (*this)(env.state());         // 玩家按当前策略选动作
                                episode.emplace_back(env.state(), action); // 记录玩家步
                                result = env.step(action);
                            }
                            // 该庄家了，但我们不记录庄家的步，只要让庄家按固定策略行动直到终局即可
                            while (not result.done and env.state().turn == Blackjack::DEALER)
                            {
                                int action = (*this)(env.state()); // 庄家按固定策略选动作
                                result = env.step(action);
                            }
                            // 终局奖励直接取最后一步的 player_reward（±1/0），不必自己重算结算逻辑
                            double G = result.player_reward;
                            // 首次访问 MC 的标准回溯：从后往前扫轨迹，G ← γ·G + r
                            // 本问题中步间奖励 r 全为 0、γ=1，所以回溯过程中 G 恒等于终局奖励；
                            // 写成倒序累加的形式是为了与书上伪代码一致，若以后奖励结构改变也不用改这里。
                            // （玩家每次 HIT 后 player_sum 一定会变化，同一 (s,a) 一局内不会重复，
                            //   天然满足"首次访问"，无需额外判重）
                            for (auto it = episode.rbegin(); it != episode.rend(); ++it)
                            {
                                // recap: 写引用是为了直接修改 value 与计数，这样就不需要再用下标访问两次了
                                const EpisodeStep &step = *it;
                                int &count = state_action_count[step.dealer_shown][step.player_ace][step.player_sum][step.action];
                                double &q = value[step.dealer_shown][step.player_ace][step.player_sum][step.action];
                                count += 1;
                                q += (G - q) / count;
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- 打印策略表：行=玩家点数(21→11)，列=庄家明牌(A→T)；左无可用A，右有可用A ----
    // 输出形式对应书上 P100 Figure 5.2 的 π*（H=要牌，S=停牌）
    void print_policy() const
    {
        cout << setw(10) << "Player Without Ace" << "\t\t" << "Player With Usable Ace." << endl;
        for (int player_sum = 21; player_sum >= 11; --player_sum)
        {
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << ACTION_NAME[policy[dealer_shown][0][player_sum]];
            }
            cout << "\t\t\t\t";
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << ACTION_NAME[policy[dealer_shown][1][player_sum]];
            }
            cout << endl;
        }
        cout << endl;
    }
    // ---- 打印当前策略下的状态价值 V(s)=Q(s,π(s))，排版与 print_policy 相同 ----
    void print_value() const
    {
        cout << setw(40) << "Player Without Ace" << setw(20) << "\t\t\t" << "Player With Usable Ace." << endl;
        for (int player_sum = 21; player_sum >= 11; --player_sum)
        {
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][policy[dealer_shown][0][player_sum]];
            }
            cout << "\t";
            for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
            {
                cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][policy[dealer_shown][1][player_sum]];
            }
            cout << endl;
        }
        cout << endl;
    }

    // 构造函数：初始化策略为"玩家 ≥20 停牌，否则要牌"（与演示策略一致，作为迭代起点）
    BlackjackPolicyLearnableDefault()
    {
        for (int dealer_shown = 1; dealer_shown <= 10; ++dealer_shown)
        {
            for (int player_ace = 0; player_ace <= 1; ++player_ace)
            {
                for (int player_sum = 0; player_sum <= 21; ++player_sum)
                {
                    int &action = policy[dealer_shown][player_ace][player_sum];
                    action = player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
                }
            }
        }
    }

private:
    // ---- 策略表 π(s) ----
    // 11: dealer_shown (A-10);
    // 2: player_usable_ace (true/false);
    // 22: player_sum (only need to consider 11-20, because HIT when sum<11 and STICK when sum=21 are always best action.)
    // 注：实际只需关心点数 11~20——<11 要牌必不亏，=21 停牌必最优
    // 第一个维度不是 10 的原因是方便直接用 dealer_shown 作为下标（1~10），0 不用
    // 第三个维度同理，player_sum 直接用点数作为下标（0~21），0~10 不用
    // 一个 state 就是一个三元组 (dealer_shown, player_ace, player_sum)，所以 policy[s] = action
    int policy[11][2][22];
    // ---- 动作价值表 Q(s,a)：MC 估计的目标 ----
    // 11:dealer_shown; 2:player_usable_ace; 22:player_sum; 2:action (0:HIT, 1:STICK)
    // value[s][a] = Q(s,a)
    double value[11][2][22][2];
    // 每个 (s,a) 被首次访问并累计回报的次数 N(s,a)，用于增量求均值
    // state_action_count[s][a] = N(s,a)
    int state_action_count[11][2][22][2];

    // record a episode sampled (only player's steps).
    // ---- 一局轨迹中"玩家的一步"（只记录玩家步，庄家步是环境的一部分） ----
    struct EpisodeStep
    {
        // state: (dealer_shown, player_ace, player_sum)：该步的玩家观测状态
        int dealer_shown;
        int player_ace;
        int player_sum;
        // the action taken at state：在该状态采取的动作
        int action;
        EpisodeStep(const Blackjack::State &state, int action)
        {
            dealer_shown = state.dealer_shown;
            player_ace = int(state.player_ace);
            player_sum = state.player_sum;
            this->action = action;
        }
        EpisodeStep() {}
    };
    // 当前采样到的一局轨迹（由 update_value 维护）
    vector<EpisodeStep> episode;

    // 清零价值表与计数表（每轮策略评估前调用）
    void set_value_initial()
    {
        memset(value, 0, sizeof(value));
        memset(state_action_count, 0, sizeof(state_action_count));
    }
};

// ---- main：当前是"固定策略演示"——无限开新局，每步打印，每秒一局 ----
// 作业要求改成：用 BlackjackPolicyLearnableDefault 反复
//   update_value（评估）→ update_policy（改进） 若干轮后 print_policy 输出最终策略
// Demonstrative play of player & dealer with default policy
int main()
{
    Blackjack env(false); // 训练阶段不打印逐步细节
    BlackjackPolicyLearnableDefault learnable_policy;
    // 蒙特卡洛 ES 策略迭代：反复 "评估 → 改进"，直到策略稳定
    const int ROUNDS = 100;       // 策略迭代轮数
    const int N_PER_START = 1000; // 每轮中每个起点（初始状态 × 首动作）模拟的局数
    for (int round = 1; round <= ROUNDS; ++round)
    {
        learnable_policy.update_value(env, N_PER_START); // 策略评估：MC 估计当前策略的 Q
        learnable_policy.update_policy();                // 策略改进：π(s) ← argmax_a Q(s,a)
        if (round % 20 == 0)                             // 每 20 轮观察一次策略表的变化
        {
            cout << "== round " << round << " ==" << endl;
            learnable_policy.print_policy();
        }
    }
    learnable_policy.print_value();  // 输出最终策略下的状态价值表
    learnable_policy.print_policy(); // 输出最终策略表（对照书上 P100 Figure 5.2）
    bool done = false;
    env.reset();        // 随机开新局
    env.verbose = true; // 演示时打印每步细节
    // 单局循环：观测状态 → 策略出动作 → 环境推进一步，直到终局
    while (not done)
    {
        Blackjack::State state = env.state();
        int action = learnable_policy(state);
        Blackjack::StepResult result = env.step(action);
        done = result.done;
    }
    cout << endl;
    this_thread::sleep_for(chrono::milliseconds(1000)); // 间隔 1 秒再开下一局
    return 0;
}