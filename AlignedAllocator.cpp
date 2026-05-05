#include <cstdlib>
#include <utility> // ova biblioteka je potrebna za std::move i std::pair
#include <type_traits> // ova biblioteka je potrebna za std::is_trivially_copyable 




namespace hpc{

template <typename T, std::size_t Alignment = 64> // primer : double , po 64 bita poravnanje
class AlignedAllocator{

public: 
    

    static_assert(Alignment >= alignof(T), "alligment premali za tip T"); // compile-time ako je false, ispisuje tekst i ne ide dalje
    static_assert((Alignment & (Alignment-1)) == 0, "Alignment mora biti stepen dvojke");
    
    using value_type = T;

    template<typename U>
    struct rebind{
        using other = AlignedAllocator<U, Alignment>; }; 
        
        
    // [[]] obavestenja za kompajler, ne odbacuj ako neko ne sacuva pointer koji mu se vrati
    [[nodiscard]] T* allocate(std::size_t n){ // koliko zelimo memorije da alociramo

        if(n==0) return nullptr;

        std::size_t bytes = n * sizeof(T); // npr T je double i zelimo 10 mesta to je 8byte * 10 = 80B
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
    
    //interno koriscenje allocator
    using Alloc = AlignedAllocator<T,Alignment>;
    
    T* data_ptr = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t stride_ = 0;
    
public:
    static std::size_t calculate_stride(std::size_t cols){

        std::size_t row_bytes = cols * sizeof(T); // npr double koji ima 4 kolone.  4*8 = 32B
        //moramo zaokruziti na 64B
        std::size_t aligned_row_bytes = (row_bytes + Alignment - 1) & ~(Alignment-1);
        return aligned_row_bytes / sizeof(T);  // zasto delimo? zato sto nam treba broj ELEMENATA u redu, DO SADA JE BILO U BAJTOVIMA. Ako imamo 32B po redu, a svaki element je 8B, onda nam treba 4 elementa po redu da bismo imali 32B. Ako imamo 40B po redu, onda nam treba 5 elemenata po redu da bismo imali 40B. Dakle delimo sa sizeof(T) da bismo dobili broj elemenata po redu.
    }

    Tensor(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols){

        stride_ = calculate_stride(cols);
        data_ptr = Alloc{}.allocate(rows_ * stride_); // alokacija memorije za ceo tensor, ali sa stride-om koji je veci od broja kolona, da bismo imali poravnanje
    }

    ~Tensor(){
        if (data_ptr) Alloc{}.deallocate(data_ptr, rows_ * stride_);
    }

    // Zabranjeno skriveno kopiranje
    Tensor(const Tensor&) = delete; // zabranjujemo copy konstruktor, jer ne zelimo da se tensor kopira, vec samo da se move-uje. Ako neko pokuša da kopira tensor, dobiće grešku na kompajliranju. 
    Tensor& operator=(const Tensor&) = delete; // zabranjujemo copy assignment operator, jer ne zelimo da se tensor kopira, vec samo da se move-uje. Ako neko pokuša da kopira tensor, dobiće grešku na kompajliranju.

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
        assert(r<rows_ && c<cols_); // provera da li su indeksi unutar granica, ako nisu, program će se srušiti sa porukom o grešci. 
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