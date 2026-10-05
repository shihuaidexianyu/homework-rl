#include <ctime>     // 提供 time() 用于随机数种子
#include <random>    // 提供泊松分布、均匀分布等随机数生成工具
#include <utility>   // 提供 std::pair 和 std::make_pair
#include <iostream>  // 提供 cout 等标准输入输出
#include <algorithm> // 提供 max、min 函数
#include <cmath>     // 提供 exp、log 和 lgamma 函数

using namespace std;

// JackCarRental 类：实现经典的 "Jack 汽车租赁" 强化学习环境
// （Sutton & Barto《强化学习》第 4 章例题 4.2）
//
// 问题描述：
//   Jack 经营两个租车点（地点 1 和 2），每个地点最多停 20 辆车。
//   每天结束时，Jack 可以在两个地点之间调配车辆（最多 5 辆），
//   每辆车的搬运费用为 2 元。
//   每个地点的租车请求和还车数量均服从泊松分布：
//     地点 1：请求均值 3，还车均值 3
//     地点 2：请求均值 4，还车均值 2
//   每租出一辆车收入 10 元。
//
// 状态：(car_1, car_2)，即两个地点各自的车辆数
// 动作：action ∈ [-5, 5]，正数表示从地点 1 调往地点 2 的车辆数，
//       负数表示从地点 2 调往地点 1 的车辆数
class JackCarRental
{
    // 环境常量：车辆上限与单次调配上限
    static const int
        MAX_CAR_1,  // 地点 1 最多可停放 20 辆车
        MAX_CAR_2,  // 地点 2 最多可停放 20 辆车
        MOVE_LIMIT; // 每晚最多可调配 5 辆车
    // 环境常量：费用、租金与各泊松分布的均值
    static const double
        MOVE_COST,      // 每调配一辆车的费用（2 元）
        RENT_PRICE,     // 每租出一辆车的收入（10 元）
        MEAN_REQUEST_1, // 地点 1 租车请求的泊松均值（3）
        MEAN_REQUEST_2, // 地点 2 租车请求的泊松均值（4）
        MEAN_RETURN_1,  // 地点 1 还车数量的泊松均值（3）
        MEAN_RETURN_2;  // 地点 2 还车数量的泊松均值（2）
    // 四个泊松分布随机数生成器（静态成员，所有实例共享）
    static poisson_distribution<int>
        request_1, request_2, return_1, return_2;

public:
    // 状态类型：用 (car_1, car_2) 数对表示两个地点的车辆数
    typedef pair<int, int> State;

    bool verbose; // 是否输出详细的调试信息

    // 获取当前状态
    State state()
    {
        return make_pair(car_1, car_2);
    }

    // 设置当前状态（直接修改两个地点的车辆数）
    void set_state(int car_1, int car_2)
    {
        if (verbose)
        {
            cout << "State set to (" << car_1 << ", " << car_2 << ")" << endl;
        }
        this->car_1 = car_1;
        this->car_2 = car_2;
    }

    // 重置环境：天数归零，两个地点的车辆数都置为 0
    void reset()
    {
        if (verbose)
        {
            cout << "Environment reset." << endl;
        }
        day = 0;
        car_1 = 0;
        car_2 = 0;
    }

    // 执行一步动作（代表经过一天），返回 <新状态, 奖励>
    pair<State, double> step(int action)
    {
        day++;
        if (verbose)
        {
            cout << "\nDay: " << day
                 << " State: (" << car_1 << ", " << car_2 << ")" << endl;
        }
        double reward = state_transition(action); // 进行状态转移并获得奖励
        if (verbose)
        {
            cout << "\tReward: " << reward << endl;
        }
        return make_pair(state(), reward);
    }

    // 随机采样一个合法动作（均匀随机）
    // 合法范围需保证调配后两个地点的车辆数不为负：
    //   下限 = max(-car_2, -5)，即最多从地点 2 调出 car_2 辆
    //   上限 = min(car_1, 5)，即最多从地点 1 调出 car_1 辆
    int sample_action()
    {
        int action_low = max(-car_2, -MOVE_LIMIT);
        int action_high = min(car_1, MOVE_LIMIT);
        uniform_int_distribution<int> random_action(action_low, action_high);
        int action = random_action(e);
        if (verbose)
        {
            cout << "\tAction " << action
                 << " sampled from uniform[" << action_low << ", " << action_high << "]" << endl;
        }
        return action;
    }

    // 构造函数：可指定初始车辆数与是否输出日志，随机数引擎用当前时间做种子
    JackCarRental(int car_1 = 0, int car_2 = 0, bool verbose = false)
    {
        this->day = 0;
        this->verbose = verbose;
        set_state(car_1, car_2);
        e.seed(time(nullptr));
    }

private:
    int car_1, car_2, day;   // 两个地点的车辆数、当前天数
    default_random_engine e; // 随机数引擎（供所有分布采样使用）

    // 状态转移函数：按"调配 → 租车 → 还车"的顺序模拟一天的经营，
    // 更新车辆数并返回当天净收益（租金收入 - 调配费用）
    double state_transition(int action)
    {
        // ---- 第 1 步：夜间调配车辆 ----
        // action > 0：从地点 1 调出 action 辆到地点 2
        // action < 0：从地点 2 调出 |action| 辆到地点 1
        // 用 min 保证不超过各地点的停车上限（超出部分被丢弃）
        car_1 = min(car_1 - action, MAX_CAR_1);
        car_2 = min(car_2 + action, MAX_CAR_2);
        double total_move_cost = abs(action) * MOVE_COST; // 调配总费用
        if (verbose)
        {
            cout << "\tMove: (" << -action << ", " << action
                 << "), cost: " << total_move_cost << endl;
            cout << "\tAfter movement, state: (" << car_1 << ", " << car_2 << ")" << endl;
        }

        // ---- 第 2 步：白天处理租车请求 ----
        int req_1 = this->request_1(e); // 地点 1 的租车请求数（泊松采样）
        int req_2 = request_2(e);       // 地点 2 的租车请求数（泊松采样）
        if (verbose)
        {
            cout << "\tRental request: (" << req_1 << ", " << req_2 << ")" << endl;
        }
        // 实际租出的车辆数受现有车辆数限制（有车才能租）
        int rent_1 = min(car_1, req_1);
        int rent_2 = min(car_2, req_2);
        double total_income = (rent_1 + rent_2) * RENT_PRICE; // 租金总收入
        car_1 -= rent_1;                                      // 扣除租出的车辆
        car_2 -= rent_2;
        if (verbose)
        {
            cout << "\tRent: (" << rent_1 << ", " << rent_2
                 << "), income: " << total_income << endl;
            cout << "\tAfter rent, state: (" << car_1 << ", " << car_2 << ")" << endl;
        }

        // ---- 第 3 步：顾客还车 ----
        int ret_1 = return_1(e); // 地点 1 的还车数（泊松采样）
        int ret_2 = return_2(e); // 地点 2 的还车数（泊松采样）
        if (verbose)
        {
            cout << "\tCars to return: (" << ret_1 << ", " << ret_2 << ")" << endl;
        }
        // 还车后车辆数不能超过停车上限（超出部分被丢弃）
        car_1 = min(car_1 + ret_1, MAX_CAR_1);
        car_2 = min(car_2 + ret_2, MAX_CAR_2);
        if (verbose)
        {
            cout << "\tAfter return, state: (" << car_1 << ", " << car_2 << ")" << endl;
        }

        // 当天净收益 = 租金收入 - 调配费用
        return total_income - total_move_cost;
    }
};

// 静态常量成员的定义（类外初始化）
const int
    JackCarRental::MAX_CAR_1 = 20, // 地点 1 停车上限
    JackCarRental::MAX_CAR_2 = 20, // 地点 2 停车上限
    JackCarRental::MOVE_LIMIT = 5; // 每晚调配上限
const double
    JackCarRental::MOVE_COST = 2.0,      // 单车调配费用
    JackCarRental::RENT_PRICE = 10.0,    // 单车租金
    JackCarRental::MEAN_REQUEST_1 = 3.0, // 地点 1 请求均值
    JackCarRental::MEAN_REQUEST_2 = 4.0, // 地点 2 请求均值
    JackCarRental::MEAN_RETURN_1 = 3.0,  // 地点 1 还车均值
    JackCarRental::MEAN_RETURN_2 = 2.0;  // 地点 2 还车均值
// 静态泊松分布生成器的定义（用各自的均值参数初始化）
poisson_distribution<int>
    JackCarRental::request_1(JackCarRental::MEAN_REQUEST_1),
    JackCarRental::request_2(JackCarRental::MEAN_REQUEST_2),
    JackCarRental::return_1(JackCarRental::MEAN_RETURN_1),
    JackCarRental::return_2(JackCarRental::MEAN_RETURN_2);

class RentPolicy
{
public:
    int policy[21][21]{0};
    double v_table[21][21]{0.0};
    void print_policy()
    {
        for (int i = 0; i < 21; ++i)
        {
            for (int j = 0; j < 21; ++j)
            {
                cout << policy[i][j] << " ";
            }
            cout << endl;
        }
    }

    // 计算泊松随机变量 X 取 count 的概率：P(X = count)。
    double poisson_probability(double mean, int count) const
    {
        if (mean < 0.0 || count < 0)
        {
            return 0.0;
        }

        // mean = 0 时不能计算 log(0)，而 X 必定等于 0。
        if (mean == 0.0)
        {
            return count == 0 ? 1.0 : 0.0;
        }

        // 使用对数形式计算，避免 mean^count 和 count! 过大。
        return exp(-mean + count * log(mean) - lgamma(count + 1.0));
    }

    /*
        计算泊松分布右尾概率 P(X >= lower_bound)。

        租车库存降到 0、或还车库存达到 20 时，需要把多个请求/还车数量
        合并到同一个边界状态，因此使用右尾概率。
    */
    double poisson_tail_probability(double mean, int lower_bound) const
    {
        if (lower_bound <= 0)
        {
            return 1.0;
        }

        double probability_below_lower_bound = 0.0;
        for (int count = 0; count < lower_bound; ++count)
        {
            probability_below_lower_bound +=
                poisson_probability(mean, count);
        }
        return max(0.0, 1.0 - probability_below_lower_bound);
    }

    /*
        计算一个地点的库存转移概率：

            给定操作前库存 current_inventory，
            经过一次租车或还车后变为 next_inventory 的概率。

        租车：
            next = max(current - request, 0)
        还车：
            next = min(current + returned, 20)

        当库存触及 0 或 20 时，多个请求/还车数量会被截断到同一个
        边界状态，因此需要使用 poisson_tail_probability。
    */
    double inventory_transition_probability(
        double mean, int current_inventory, int next_inventory, bool is_rental) const
    {
        const int max_inventory = 20;
        if (mean < 0.0 ||
            current_inventory < 0 || current_inventory > max_inventory ||
            next_inventory < 0 || next_inventory > max_inventory)
        {
            return 0.0;
        }

        if (is_rental)
        {
            if (next_inventory == 0)
            {
                return poisson_tail_probability(mean, current_inventory);
            }
            if (next_inventory > current_inventory)
            {
                return 0.0;
            }
            return poisson_probability(mean, current_inventory - next_inventory);
        }

        if (next_inventory == max_inventory)
        {
            return poisson_tail_probability(
                mean, max_inventory - current_inventory);
        }
        if (next_inventory < current_inventory)
        {
            return 0.0;
        }
        return poisson_probability(mean, next_inventory - current_inventory);
    }

    // 判断动作是否能在当前状态下执行。
    bool is_legal_action(JackCarRental::State old_state, int action) const
    {
        const int max_cars = 20;
        const int move_limit = 5;
        const int moved_car_1 = old_state.first - action;
        const int moved_car_2 = old_state.second + action;

        return action >= -move_limit && action <= move_limit &&
               moved_car_1 >= 0 && moved_car_1 <= max_cars &&
               moved_car_2 >= 0 && moved_car_2 <= max_cars;
    }

    /*
        应用确定性的调车动作。

        这个函数只负责第一阶段的调车，不进行随机采样，也不计算租车
        或还车。调用前应使用 is_legal_action 检查 action。
    */
    JackCarRental::State apply_movement(
        JackCarRental::State old_state, int action) const
    {
        return make_pair(old_state.first - action, old_state.second + action);
    }

    /*
        计算 Q(s, a)：

            Q(s, a) = E[r + gamma * V(s') | s, a]

        其中：
            s       是调车前状态；
            a       是调车动作；
            r       是当天租车收入减去调车成本；
            s'      是还车结束后的最终状态。

        调车是确定性的，因此先得到 moved_state。
        随后有两层随机过程：
            1. 枚举租车后的中间状态 rental_state；
            2. 从 rental_state 出发，枚举还车后的 next_state。

        两个地点相互独立，所以每个阶段的联合概率可以分别相乘。
    */
    double expected_action_value(
        JackCarRental::State old_state, int action, double gamma) const
    {
        const int max_cars = 20;
        const double move_cost = 2.0;
        const double rent_price = 10.0;

        if (gamma < 0.0 || gamma > 1.0 ||
            !is_legal_action(old_state, action))
        {
            return 0.0;
        }
        // 先调车
        const JackCarRental::State moved_state =
            apply_movement(old_state, action);
        const double movement_cost = abs(action) * move_cost;
        double expected_value = 0.0;

        // 第一层：枚举两个地点租车结束后的中间状态。
        for (int rental_1 = 0; rental_1 <= max_cars; ++rental_1)
        {
            const double rental_probability_1 =
                inventory_transition_probability(
                    3.0, moved_state.first, rental_1, true);

            for (int rental_2 = 0; rental_2 <= max_cars; ++rental_2)
            {
                const double rental_probability_2 =
                    inventory_transition_probability(
                        4.0, moved_state.second, rental_2, true);
                const double rental_probability =
                    rental_probability_1 * rental_probability_2; // 联合概率

                if (rental_probability == 0.0)
                {
                    continue;
                }

                const double rental_income =
                    ((moved_state.first - rental_1) +
                     (moved_state.second - rental_2)) *
                    rent_price;

                // 第二层：从租车后的中间状态枚举还车后的最终状态。
                for (int next_1 = 0; next_1 <= max_cars; ++next_1)
                {
                    const double return_probability_1 =
                        inventory_transition_probability(
                            3.0, rental_1, next_1, false);

                    for (int next_2 = 0; next_2 <= max_cars; ++next_2)
                    {
                        const double return_probability_2 =
                            inventory_transition_probability(
                                2.0, rental_2, next_2, false);
                        const double transition_probability =
                            rental_probability *
                            return_probability_1 *
                            return_probability_2; // 依旧联合概率

                        expected_value += transition_probability *
                                          (rental_income - movement_cost +
                                           gamma * v_table[next_1][next_2]);
                    }
                }
            }
        }

        return expected_value;
    }

    // 按当前 policy 反复更新 V，直到所有状态的价值变化都小于 theta。
    void policy_evaluate(double gamma = 0.9, double theta = 1e-4)
    {
        while (true)
        {
            double delta = 0.0;
            for (int i = 0; i < 21; ++i)
            {
                for (int j = 0; j < 21; ++j)
                {
                    const JackCarRental::State current_state =
                        make_pair(i, j);
                    const double old_value = v_table[i][j];
                    const double new_value =
                        expected_action_value(
                            current_state, policy[i][j], gamma);
                    v_table[i][j] = new_value;
                    delta = max(delta, abs(old_value - new_value));
                }
            }
            if (delta < theta)
            {
                break;
            }
        }
        cout << "evaluate complete." << endl;
    }

    bool policy_improve()
    {
        bool policy_stable = true;
        for (int i = 0; i < 21; ++i)
        {
            for (int j = 0; j < 21; ++j)
            {
                int old_action = policy[i][j];
                double old_action_value = expected_action_value(
                    make_pair(i, j), old_action, 0.9);
                for (int a = -5; a <= 5; ++a)
                {
                    if (is_legal_action(make_pair(i, j), a))
                    {
                        const double action_value = expected_action_value(
                            make_pair(i, j), a, 0.9);
                        if (action_value > old_action_value)
                        {
                            policy_stable = false;
                            policy[i][j] = a;
                            continue;
                        }
                    }
                }
            }
        }
        cout << "improve complete." << endl;
        return policy_stable;
    }
};

#include <chrono> // 提供时间相关工具
#include <thread> // 提供 sleep_for 线程休眠函数

// 主函数：创建环境并随机执行动作，用于直观演示环境的运行
int main()
{
    // 创建 JackCarRental 环境，初始状态 (0, 0)，开启详细日志输出
    JackCarRental env(0, 0, true);
    double gamma = 0.9;
    double theta = 1e-4;

    RentPolicy dp = RentPolicy();
    while (true)
    {
        dp.policy_evaluate();
        if (dp.policy_improve())
        {
            break;
        }
    }
    dp.print_policy();
    return 0;
}