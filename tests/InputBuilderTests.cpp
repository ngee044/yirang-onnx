#include "InputBuilder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace YirangOnnx;

namespace
{
	auto make_value_info(void) -> ValueInfo
	{
		ValueInfo info;
		info.name_ = "X";
		info.data_type_ = "FLOAT";
		info.elem_type_ = 1;
		info.shape_ = { "1", "N" };
		return info;
	}
} // namespace

TEST(InputBuilderTest, ResolvesShapeWithOverrides)
{
	auto [resolved, error] = resolve_input_shape(make_value_info(), { { "N", 3 } });
	ASSERT_TRUE(resolved.has_value()) << error.value_or("");
	ASSERT_EQ(resolved->dims_.size(), 2u);
	EXPECT_EQ(resolved->dims_[0], 1);
	EXPECT_EQ(resolved->dims_[1], 3);
	EXPECT_TRUE(resolved->notes_.empty());
}

TEST(InputBuilderTest, DefaultsSymbolicDimensionsToOne)
{
	auto [resolved, error] = resolve_input_shape(make_value_info(), {});
	ASSERT_TRUE(resolved.has_value());
	ASSERT_EQ(resolved->dims_.size(), 2u);
	EXPECT_EQ(resolved->dims_[1], 1);
	ASSERT_EQ(resolved->notes_.size(), 1u);
	EXPECT_NE(resolved->notes_[0].find("'N'"), std::string::npos);
}

TEST(InputBuilderTest, RandomTensorIsDeterministicWithSeed)
{
	const std::vector<int64_t> shape = { 2, 3 };
	auto [first, first_error] = make_random_tensor("X", 1, shape, 42);
	auto [second, second_error] = make_random_tensor("X", 1, shape, 42);
	ASSERT_TRUE(first.has_value()) << first_error.value_or("");
	ASSERT_TRUE(second.has_value());

	EXPECT_EQ(first->name_, "X");
	EXPECT_EQ(first->elem_type_, 1);
	EXPECT_EQ(first->shape_, shape);
	EXPECT_EQ(first->data_.size(), 6u * sizeof(float));
	EXPECT_EQ(first->data_, second->data_);
}

TEST(InputBuilderTest, RandomTensorSupportsIntegerTypes)
{
	auto [tensor, error] = make_random_tensor("ids", 7, { 4 }, 7);
	ASSERT_TRUE(tensor.has_value()) << error.value_or("");
	EXPECT_EQ(tensor->data_.size(), 4u * sizeof(int64_t));
}

TEST(InputBuilderTest, RejectsUnsupportedRandomType)
{
	auto [tensor, error] = make_random_tensor("X", 8, { 1 }, 0);
	EXPECT_FALSE(tensor.has_value());
	ASSERT_TRUE(error.has_value());
	EXPECT_NE(error->find("STRING"), std::string::npos);
}
