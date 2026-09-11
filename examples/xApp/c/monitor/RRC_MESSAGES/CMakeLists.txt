# SPDX-License-Identifier: LicenseRef-CSSL-1.0

set(NR_RRC_VERSION 17 3 0)
include(macros.cmake)
make_version(NR_RRC_cc ${NR_RRC_VERSION})
string(REPLACE ";" "." NR_RRC_RELEASE "${NR_RRC_VERSION}")

if(NR_RRC_RELEASE VERSION_EQUAL "17.3.0")
  include(nr-rrc-17.3.0.cmake)
else()
  message(FATAL_ERROR "unknown NR_RRC_RELEASE ${NR_RRC_RELEASE}")
endif()

find_program(ASN1C_EXEC_PATH asn1c HINTS /opt/asn1c/bin)
set(ASN1C_EXEC ${ASN1C_EXEC_PATH} CACHE FILEPATH "path to asn1c executable")
run_asn1c(${CMAKE_CURRENT_SOURCE_DIR}/${NR_RRC_GRAMMAR}
          "NR_"
          OUTPUT ${nr_rrc_source} ${nr_rrc_headers}
          # NOTE (DJCSR): -no-gen-CBOR added. Newer mouse07410/asn1c emits CBOR codec entries in
          # the type descriptors, but nr-rrc-17.3.0.cmake's explicit source list predates CBOR and
          # never compiles the generated *_cbor.c files, producing undefined *_encode_cbor refs at
          # link time. Disabling CBOR matches how BER/JER/OER are already switched off here.
          OPTIONS -pdu=all -fcompound-names -gen-UPER -no-gen-BER -no-gen-JER -no-gen-OER -no-gen-CBOR -gen-APER -no-gen-example -findirect-choice
          COMMENT "NR RRC source files"
)

add_library(asn1_nr_rrc_hdrs INTERFACE ${nr_rrc_headers})
target_include_directories(asn1_nr_rrc_hdrs INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")

# NOTE (DJCSR): newer mouse07410/asn1c split these runtime helpers into their own files,
# which nr-rrc-17.3.0.cmake's fixed source list does not know about. Without them the link
# fails with undefined asn_check_INTEGER_range / OPEN_TYPE_print_member / ENUMERATED_*.
# They are produced by run_asn1c above, so mark them GENERATED and compile them in.
set(nr_rrc_extra_source
    "${CMAKE_CURRENT_BINARY_DIR}/asn_constraint_value.c"
    "${CMAKE_CURRENT_BINARY_DIR}/OPEN_TYPE_print.c"
    "${CMAKE_CURRENT_BINARY_DIR}/ENUMERATED.c"
    "${CMAKE_CURRENT_BINARY_DIR}/ENUMERATED_aper.c"
    "${CMAKE_CURRENT_BINARY_DIR}/ENUMERATED_uper.c")
set_source_files_properties(${nr_rrc_extra_source} PROPERTIES GENERATED TRUE)

add_library(asn1_nr_rrc ${nr_rrc_source} ${nr_rrc_extra_source} ${nr_rrc_headers})
target_link_libraries(asn1_nr_rrc PUBLIC asn1_nr_rrc_hdrs)
target_compile_options(asn1_nr_rrc
                         PRIVATE -DASN_DISABLE_OER_SUPPORT -w
                         PUBLIC -DNR_RRC_VERSION=${NR_RRC_RELEASE})

