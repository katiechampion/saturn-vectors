package saturn.exu

import chisel3._
import chisel3.util._
import chiseltest._
import org.chipsalliance.cde.config._
import freechips.rocketchip.tile._
import org.scalatest.funspec.AnyFunSpec
import scala.io.Source

class BulkNormalizerTest extends AnyFunSpec with ChiselScalatestTester {

    describe("Bulk Normalizer") {
        val data_source = Source.fromFile("./generators/saturn/benchmarks/vec-bdot-fp/gen_data_unit_test/data.txt")
        val data = data_source.getLines().map(_.split(',').map(Integer.parseInt(_, 16))).toSeq

        it("bulk normalize") {
            test(new BulkNormalizerMultiplier(FType.BF16, 8)(Parameters.empty)) { dut =>
                dut.io.in_a.zip(data(0)).foreach { case (in, dat) => in.poke(dat.U) }
                dut.io.in_b.zip(data(1)).foreach { case (in, dat) => in.poke(dat.U) }

                dut.io.out.any_nan.expect(false.B)
                dut.io.out.any_inf.expect(false.B)
                dut.io.out.any_pos_inf.expect(false.B)
                dut.io.out.any_neg_inf.expect(false.B)
                
                println(dut.io.out.prod_signs.peek)
                println(dut.io.out.prod_exps.peek)
                println(dut.io.out.prod_sigs.peek)
            }
        }
    }
}