#include <utility>  // 提供 std::pair 和 std::make_pair
#include <cstdlib>  // 提供 rand() 随机数函数
#include <iostream> // 提供 cout 等标准输入输出

using namespace std;

// GridWorld 类：实现一个 5x5 的网格世界环境（强化学习经典示例）
// 坐标范围：x ∈ [0, 4]，y ∈ [0, 4]
// 特殊状态：
//   (1, 0)：任意动作后跳转到 (1, 4)，奖励 +10
//   (3, 0)：任意动作后跳转到 (3, 2)，奖励 +5
// 撞墙（越界动作）：位置不变，奖励 -1
// 其余普通移动：奖励 0
class GridWorld
{
public:
    // 定义四个动作方向常量
    static const int
        NORTH = 0,                       // 北：y 减 1
        SOUTH = 1,                       // 南：y 加 1
        EAST = 2,                        // 东：x 加 1
        WEST = 3;                        // 西：x 减 1
    static const char ACTION_NAME[][16]; // 各动作的名称（用于打印日志）

    // 状态类型：用 (x, y) 坐标对表示
    typedef pair<int, int> State;

    bool verbose; // 是否输出详细的调试信息

    // 获取当前状态（当前坐标）
    State state()
    {
        return make_pair(x, y);
    }

    // 设置当前状态（直接修改坐标）
    void set_state(int x, int y)
    {
        this->x = x;
        this->y = y;
        if (verbose)
        {
            cout << "State reset: (" << x << "," << y << ")" << endl;
        }
    }

    // 重置环境到起始状态 (0, 0)
    void reset()
    {
        set_state(0, 0);
    }

    // 执行一步动作，返回 <新状态, 奖励>
    pair<State, double> step(int action)
    {
        State old_state = state();                // 记录执行动作前的状态
        double reward = state_transition(action); // 进行状态转移并获得奖励
        if (verbose)
        {
            // 打印本步的详细信息：旧状态、动作、奖励、新状态
            cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
            cout << "Action: " << ACTION_NAME[action] << endl;
            cout << "Reward: " << reward << endl;
            cout << "New State: (" << x << "," << y << ")" << endl
                 << endl;
        }
        return make_pair(state(), reward);
    }

    // 随机采样一个动作（0~3 均匀随机）
    int sample_action()
    {
        return rand() % 4;
    }

    // 构造函数：可指定初始坐标与是否输出日志，默认从 (0, 0) 开始
    GridWorld(int x = 0, int y = 0, bool verbose = false)
    {
        set_state(x, y);
        this->verbose = verbose;
    }

private:
    int x, y; // 当前位置的横、纵坐标

    // 状态转移函数：根据动作更新位置，并返回奖励
    double state_transition(int action)
    {
        // 特殊状态 A：(1, 0)，无论执行什么动作都跳转到 (1, 4)，奖励 +10
        if (state() == make_pair(1, 0))
        {
            x = 1;
            y = 4;
            return 10.0;
        }
        // 特殊状态 B：(3, 0)，无论执行什么动作都跳转到 (3, 2)，奖励 +5
        if (state() == make_pair(3, 0))
        {
            x = 3;
            y = 2;
            return 5.0;
        }
        // 撞墙判断：动作会导致越界时，位置不变，奖励 -1
        // 注意 and/or 的优先级：and 高于 or，等价于四个条件各自相与后再相或
        if (action == NORTH and y == 0 or // 在最上边界仍向北走
            action == SOUTH and y == 4 or // 在最下边界仍向南走
            action == EAST and x == 4 or  // 在最右边界仍向东走
            action == WEST and x == 0)    // 在最左边界仍向西走
        {
            return -1.0;
        }
        // 正常移动：根据动作方向更新坐标
        switch (action)
        {
        case NORTH:
            y--; // 向北：y 减 1
            break;
        case SOUTH:
            y++; // 向南：y 加 1
            break;
        case EAST:
            x++; // 向东：x 加 1
            break;
        case WEST:
            x--; // 向西：x 减 1
            break;
        }
        return 0.0; // 普通移动奖励为 0
    }
};

// 动作名称数组的定义，用于日志输出
const char GridWorld::ACTION_NAME[][16] = {"NORTH(0,-1)", "SOUTH(0,1)", "EAST:(1,0)", "WEST:(-1,0)"};

#include <chrono> // 提供时间相关工具
#include <thread> // 提供 sleep_for 线程休眠函数

// 主函数：创建环境并随机执行动作，用于直观演示环境的运行
int main()
{
    // 创建 GridWorld 环境，初始状态 (0, 0)，开启详细日志输出
    GridWorld env = GridWorld(0, 0, true);
    while (true)
    {
        int action = env.sample_action();                   // 随机选择一个动作
        auto state_reward = env.step(action);               // 执行动作，获取新状态和奖励
        this_thread::sleep_for(chrono::milliseconds(1000)); // 每步暂停 1 秒，便于观察
    }
    return 0;
}