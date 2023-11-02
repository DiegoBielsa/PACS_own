#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template<typename T>
class threadsafe_queue
{
  private:
      std::mutex mut;
      std::queue<T> data_queue;
      std::condition_variable data_cond;

  public:
    threadsafe_queue() {}

    threadsafe_queue(const threadsafe_queue& other)
    {
        std::lock_guard<std::mutex> lk(other.mut);
	    data_queue = other.data_queue;
    }

    threadsafe_queue& operator=(const threadsafe_queue&) = delete;

    void push(T new_value)
    {
        
        std::lock_guard<std::mutex> lk(this->mut);
        data_queue.push(new_value);
        data_cond.notify_one();
    }

    bool try_pop(T& value)
    {
	    if (this->empty()) return false;
        value = data_queue.front();
        return true;
    }

    void wait_and_pop(T& value)
    {
	    std::unique_lock<std::mutex> lk(this->mut);
        data_cond.wait(lk, try_pop(value));
        data_queue.pop();
        lk.unlock();
    }

    std::shared_ptr<T> wait_and_pop()
    {
	    std::unique_lock<std::mutex> lk(this->mut);
        T value;
        data_cond.wait(lk, try_pop(value));
        std::shared_ptr<T> value_ptr = std::make_shared<T>(value);
        data_queue.pop();
        lk.unlock();
        return value_ptr;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lk(this->mut);
	    return data_queue.empty();
    }
};
