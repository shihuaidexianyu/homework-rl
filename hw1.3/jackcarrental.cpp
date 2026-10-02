#include <ctime>     // 提供 time() 用于随机数种子
#include <random>    // 提供泊松分布、均匀分布等随机数生成工具
#include <utility>   // 提供 std::pair 和 std::make_pair
#include <iostream>  // 提供 cout 等标准输入输出
#include <algorithm> // 提供 max、min 函数

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

#include <chrono> // 提供时间相关工具
#include <thread> // 提供 sleep_for 线程休眠函数

// 主函数：创建环境并随机执行动作，用于直观演示环境的运行
int main()
{
    // 创建 JackCarRental 环境，初始状态 (0, 0)，开启详细日志输出
    JackCarRental env(0, 0, true);
    while (true)
    {
        int action = env.sample_action();                   // 随机选择一个合法动作
        env.step(action);                                   // 执行动作，模拟一天的经营
        this_thread::sleep_for(chrono::milliseconds(1000)); // 每天暂停 1 秒，便于观察
    }
    return 0;
}