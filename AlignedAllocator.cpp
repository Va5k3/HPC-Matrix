#include <cstdlib>
#include <utility> 
#include <type_traits> 



namespace hpc{

template <typename T, std::size_t Alignment = 64> 
class AlignedAllocator{

public: 
    

    static_assert(Alignment >= alignof(T), "alligment premali za tip T"); 
    static_assert((Alignment & (Alignment-1)) == 0, "Alignment mora biti stepen dvojke");
    
    using value_type = T;

    template<typename U>
    struct rebind{
        using other = AlignedAllocator<U, Alignment>; }; 
        
        
    
    [[nodiscard]] T* allocate(std::size_t n){ 

        if(n==0) return nullptr;

        std::size_t bytes = n * sizeof(T); 
        std::size_t aligned_byte = (bytes + Alignment - 1) & ~(Alignment - 1);

        void* ptr = std::aligned_alloc(Alignment, aligned_byte);
        if(!ptr) throw std::bad_alloc{};
        
        return static_cast<T*>(ptr); 
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }


};

template<typename T, std::size_t Alignment = 64>
class Tensor{
    static_assert(std::is_trivially_copyable<T>::value, "T mora biti trivially_copyable");
    
   
    using Alloc = AlignedAllocator<T,Alignment>;
    
    T* data_ptr = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t stride_ = 0;
    
public:
    static std::size_t calculate_stride(std::size_t cols){

        std::size_t row_bytes = cols * sizeof(T); 
        
        std::size_t aligned_row_bytes = (row_bytes + Alignment - 1) & ~(Alignment-1);
        return aligned_row_bytes / sizeof(T);  
    }

    Tensor(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols){

        stride_ = calculate_stride(cols);
        data_ptr = Alloc{}.allocate(rows_ * stride_); 
    }

    ~Tensor(){
        if (data_ptr) Alloc{}.deallocate(data_ptr, rows_ * stride_);
    }

   
    Tensor(const Tensor&) = delete; 
    Tensor& operator=(const Tensor&) = delete; 

    Tensor(Tensor&& other) noexcept 
        : data_ptr(other.data_ptr), rows_(other.rows_), cols_(other.cols_), stride_(other.stride_){
            other.data_ptr = nullptr;
        }

    Tensor& operator=(Tensor&& other) noexcept {
        if(this!=&other){
            if(data_ptr) Alloc{}.deallocate(data_ptr,rows_*stride_);
            data_ptr = other.data_ptr;
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            other.data_ptr = nullptr;
        }
        return *this;
    }

    T& operator()(std::size_t r, std::size_t c){
        assert(r<rows_ && c<cols_); 
        return data_ptr[r*stride_ + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        return data_ptr[r*stride_ + c];
    }

    T* data() {return data_ptr;}

    auto shape() const {
        return std::make_pair(rows_,cols_);
    }
};



}
