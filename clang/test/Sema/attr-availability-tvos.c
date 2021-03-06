// RUN: %clang_cc1 "-triple" "x86_64-apple-tvos13.0" -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-apple-tvos13.0" -DUSE_VERSION_MAP -isysroot %S/Inputs/AppleTVOS15.0.sdk -fsyntax-only -verify %s

void f0(int) __attribute__((availability(tvos,introduced=12.0,deprecated=12.1))); // expected-note {{'f0' has been explicitly marked deprecated here}}
void f1(int) __attribute__((availability(tvos,introduced=12.1)));
void f2(int) __attribute__((availability(tvos,introduced=12.0,deprecated=13.0))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(tvos,introduced=13.0)));
void f4(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(tvos,introduced=12.0,deprecated=12.1,obsoleted=13.0))); // expected-note{{explicitly marked unavailable}}

void f5(int) __attribute__((availability(tvos,introduced=12.0))) __attribute__((availability(tvos,deprecated=13.0))); // expected-note {{'f5' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(tvos,deprecated=13.0))); // expected-note {{'f6' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(tvos,introduced=12.0)));

void test() {
  f0(0); // expected-warning{{'f0' is deprecated: first deprecated in tvOS 12.1}}
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in tvOS 13.0}}
  f3(0);
  f4(0); // expected-error{{f4' is unavailable: obsoleted in tvOS 13.0}}
  f5(0); // expected-warning{{'f5' is deprecated: first deprecated in tvOS 13.0}}
  f6(0); // expected-warning{{'f6' is deprecated: first deprecated in tvOS 13.0}}
}

// Anything iOS later than 13 does not apply to tvOS.
void f9(int) __attribute__((availability(ios,introduced=12.0,deprecated=19.0)));

void test_transcribed_availability() {
  f9(0);
}

__attribute__((availability(ios,introduced=19_0,deprecated=19_0,message="" ))) // expected-warning 2{{availability does not match previous declaration}}
__attribute__((availability(ios,introduced=17_0)))                             // expected-note 2{{previous attribute is here}}
void f10(int);

// Test tvOS specific attributes.
void f0_tvos(int) __attribute__((availability(tvos,introduced=12.0,deprecated=12.1))); // expected-note {{'f0_tvos' has been explicitly marked deprecated here}}
void f1_tvos(int) __attribute__((availability(tvos,introduced=12.1)));
void f2_tvos(int) __attribute__((availability(tvOS,introduced=12.0,deprecated=13.0))); // expected-note {{'f2_tvos' has been explicitly marked deprecated here}}
void f3_tvos(int) __attribute__((availability(tvos,introduced=13.0)));
void f4_tvos(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(tvos,introduced=12.0,deprecated=12.1,obsoleted=13.0))); // expected-note{{explicitly marked unavailable}}
void f5_tvos(int) __attribute__((availability(tvos,introduced=12.0))) __attribute__((availability(ios,deprecated=13.0)));
void f5_attr_reversed_tvos(int) __attribute__((availability(ios, deprecated=13.0))) __attribute__((availability(tvos,introduced=12.0)));
void f5b_tvos(int) __attribute__((availability(tvos,introduced=12.0))) __attribute__((availability(tvos,deprecated=13.0))); // expected-note {{'f5b_tvos' has been explicitly marked deprecated here}}
void f5c_tvos(int) __attribute__((availability(ios,introduced=12.0))) __attribute__((availability(ios,deprecated=13.0))); // expected-note {{'f5c_tvos' has been explicitly marked deprecated here}}
void f6_tvos(int) __attribute__((availability(tvos,deprecated=13.0))); // expected-note {{'f6_tvos' has been explicitly marked deprecated here}}
void f6_tvos(int) __attribute__((availability(tvOS,introduced=12.0)));

void test_tvos() {
  f0_tvos(0); // expected-warning{{'f0_tvos' is deprecated: first deprecated in tvOS 12.1}}
  f1_tvos(0);
  f2_tvos(0); // expected-warning{{'f2_tvos' is deprecated: first deprecated in tvOS 13.0}}
  f3_tvos(0);
  f4_tvos(0); // expected-error{{'f4_tvos' is unavailable: obsoleted in tvOS 13.0}}
  // We get no warning here because any explicit 'tvos' availability causes
  // the ios availability to not implicitly become 'tvos' availability.  Otherwise we'd get
  // a deprecated warning.
  f5_tvos(0); // no-warning
  f5_attr_reversed_tvos(0); // no-warning
  // We get a deprecated warning here because both attributes are explicitly 'tvos'.
  f5b_tvos(0); // expected-warning {{'f5b_tvos' is deprecated: first deprecated in tvOS 13.0}}
  // We get a deprecated warning here because both attributes are 'ios' (both get mapped to 'tvos').
  f5c_tvos(0); // expected-warning {{'f5c_tvos' is deprecated: first deprecated in tvOS 13.0}}
  f6_tvos(0); // expected-warning{{'f6_tvos' is deprecated: first deprecated in tvOS 13.0}}
}

#ifdef USE_VERSION_MAP
// iOS 9.3 corresponds to tvOS 9.2, as indicated in 'SDKSettings.json'.
void f11(int) __attribute__((availability(ios,deprecated=9.3))); // expected-note {{'f11' has been explicitly marked deprecated here}}

void testWithVersionMap() {
  f11(0); // expected-warning {{'f11' is deprecated: first deprecated in tvOS 9.2}}
}
#else
// Without VersionMap, tvOS version is inferred incorrectly as 9.3.
void f11(int) __attribute__((availability(ios,deprecated=9.3))); // expected-note {{'f11' has been explicitly marked deprecated here}}

void testWithoutVersionMap() {
  f11(0); // expected-warning {{'f11' is deprecated: first deprecated in tvOS 9.3}}
}
#endif
